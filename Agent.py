import Config
from Model import *
from Replay import ReplayTuple

import random
from chainer import serializers, optimizers
if Config.gpu:
    import cupy
from chainer import cuda

import logging


class Agent(object):

    def __init__(self, _shared, _head, _env, _replay=None, _pre_model=None):
        self.env = _env
        self.replay = _replay

        # model for train, model for target
        self.q_func, self.target_q_func = buildModel(
            _shared, _head, _pre_model=None)
        self.optimizer = optimizers.Adam()
        self.optimizer.setup(self.q_func)

    def step(self):
        if not self.env.in_game:
            logging.info('Env not in game')
            self.env.startNewGame()
            self.use_head = random.randint(0, Config.K - 1)
            logging.info('Use head: ' + str(self.use_head))

        # get current state
        cur_state = self.env.getState()
        # choose action in step
        action = self.chooseAction(self.q_func, cur_state)
        # do action and get reward
        reward = self.env.doAction(action)
        # get new state
        next_state = self.env.getState()

        logging.info('Action: ' + str(action) + '; Reward: ' + str(reward))

        # randomly decide to store tuple into pool
        if random.random() < Config.replay_p:
            # store replay_tuple into memory pool
            replay_tuple = ReplayTuple(
                cur_state, action, reward, next_state,
                # get mask for bootstrap
                np.random.binomial(1, Config.p, (Config.K)).tolist()
            )
            self.replay.push(replay_tuple)

    def train(self):
        # clear grads
        self.q_func.zerograds()

        # pull tuples from memory pool
        batch_tuples = self.replay.pull(Config.batch_size)
        if not len(batch_tuples):
            return

        # stack inputs
        cur_x = [self.env.getX(t.state) for t in batch_tuples]
        next_x = [self.env.getX(t.next_state) for t in batch_tuples]
        # merge inputs into one array
        if Config.gpu:
            # cur_x = [cupy.expand_dims(t, 0) for t in cur_x]
            cur_x = cupy.concatenate(cur_x, 0)
            # next_x = [cupy.expand_dims(t, 0) for t in next_x]
            next_x = cupy.concatenate(next_x, 0)
        else:
            # cur_x = np.stack(cur_x)
            # next_x = np.stack(next_x)
            cur_x = np.concatenate(cur_x, 0)
            next_x = np.concatenate(next_x, 0)

        # get cur outputs
        cur_output = self.QFunc(self.q_func, cur_x)
        # get next outputs, NOT target
        next_output = self.QFunc(self.q_func, next_x)
        # choose next action for each output
        next_action = [
            self.env.getBestAction(
                o.data,
                [t.next_state for t in batch_tuples]
            ) for o in next_output  # for each head in Model
        ]
        # get next outputs, target
        next_output = self.QFunc(self.target_q_func, next_x)

        # clear err of tuples
        for t in batch_tuples:
            t.err = 0.
        # store err count
        err_count_list = [0.] * len(batch_tuples)

        # compute grad's weights
        weights = np.array([t.P for t in batch_tuples], np.float32)
        if Config.gpu:
            weights = cuda.to_gpu(weights)
        if self.replay.getPoolSize():
            weights *= self.replay.getPoolSize()
        weights = weights ** -Config.beta
        weights /= weights.max()
        if Config.gpu:
            weights = cupy.expand_dims(weights, 1)
        else:
            weights = np.expand_dims(weights, 1)

        # update beta
        Config.beta = min(1, Config.beta + Config.beta_add)

        # compute grad for each head
        for k in range(Config.K):
            if Config.gpu:
                cur_output[k].grad = cupy.zeros_like(cur_output[k].data)
            else:
                cur_output[k].grad = np.zeros_like(cur_output[k].data)
            # compute grad from each tuples
            for i in range(len(batch_tuples)):
                if batch_tuples[i].mask[k]:
                    cur_action_value = \
                        cur_output[k].data[i][batch_tuples[i].action].tolist()
                    reward = batch_tuples[i].reward
                    next_action_value = \
                        next_output[k].data[i][next_action[k][i]].tolist()
                    target_value = reward
                    # if not empty position, not terminal state
                    if batch_tuples[i].next_state.in_game:
                        target_value += Config.gamma * next_action_value
                    loss = cur_action_value - target_value
                    cur_output[k].grad[i][batch_tuples[i].action] = 2 * loss
                    # count err
                    if cur_action_value:
                        batch_tuples[i].err += abs(loss / cur_action_value)
                        err_count_list[i] += 1

            # multiply weights with grad and clip
            if Config.gpu:
                cur_output[k].grad = cupy.multiply(
                    cur_output[k].grad, weights)
                cur_output[k].grad = cupy.clip(cur_output[k].grad, -1, 1)
            else:
                cur_output[k].grad = np.multiply(
                    cur_output[k].grad, weights)
                cur_output[k].grad = np.clip(cur_output[k].grad, -1, 1)
            # backward
            cur_output[k].backward()

        # adjust grads of shared
        for param in self.q_func.shared.params():
            param.grad /= Config.K

        # update params
        self.optimizer.update()

        # avg err
        for i in range(len(batch_tuples)):
            if err_count_list[i] > 0:
                batch_tuples[i].err /= err_count_list[i]

        self.replay.merge(Config.alpha)

        return np.mean([t.err for t in batch_tuples])

    def chooseAction(self, _model, _state):
        # update epsilon
        Config.epsilon = max(
            Config.epsilon_underline,
            Config.epsilon * Config.epsilon_decay
        )
        random_value = random.random()
        if random_value < Config.epsilon:
            # randomly choose
            return self.env.getRandomAction(_state)
        else:
            # use model to choose
            x_data = self.env.getX(_state)
            output = self.QFunc(_model, x_data)
            return self.env.getBestAction(output[self.use_head].data, [_state])[0]

    def QFunc(self, _model, _x_data):
        def toVariable(_data):
            if type(_data) is list:
                return [toVariable(d) for d in _data]
            else:
                return Variable(_data)
        return _model(toVariable(_x_data))

    def updateTargetQFunc(self):
        logging.info('')
        self.target_q_func.copyparams(self.q_func)

    def save(self, _step):
        filename = './models/step_' + str(_step)
        logging.info(filename)
        serializers.save_npz(filename, self.q_func)
