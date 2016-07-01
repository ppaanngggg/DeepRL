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
        self.optimizer = optimizers.RMSprop()
        self.optimizer.setup(self.q_func)

    def step(self):
        while not self.env.in_game:
            logging.info('Env not in game')
            self.env.startNewGame()
            if Config.bootstrap:
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

        logging.info('Action: ' + str(action) + '; Reward: %.3f' % (reward))

        # randomly decide to store tuple into pool
        if random.random() < Config.replay_p:
            mask = None
            if Config.bootstrap:
                mask = np.random.binomial(1, Config.p, (Config.K)).tolist()
            # store replay_tuple into memory pool
            replay_tuple = ReplayTuple(
                cur_state, action, reward, next_state,
                # get mask for bootstrap
                mask
            )
            self.replay.push(replay_tuple)

    def getInputs(self, _batch_tuples):
        # stack inputs
        cur_x = [self.env.getX(t.state) for t in _batch_tuples]
        next_x = [self.env.getX(t.next_state) for t in _batch_tuples]
        # merge inputs into one array
        if Config.gpu:
            cur_x = cupy.concatenate(cur_x, 0)
            next_x = cupy.concatenate(next_x, 0)
        else:
            cur_x = np.concatenate(cur_x, 0)
            next_x = np.concatenate(next_x, 0)
        return cur_x, next_x

    def getWeights(self, _batch_tuples):
        # compute grad's weights
        weights = np.array([t.P for t in _batch_tuples], np.float32)
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

        return weights

    def forward(self, _cur_x, _next_x, _state_list):
        # get cur outputs
        cur_output = self.QFunc(self.q_func, _cur_x)
        if Config.double_q:
            # get next outputs, NOT target
            next_output = self.QFunc(self.q_func, _next_x)
        else:
            next_output = self.QFunc(self.target_q_func, _next_x)

        if Config.bootstrap:
            # choose next action for each output
            next_action = [
                self.env.getBestAction(
                    o.data,
                    _state_list
                ) for o in next_output  # for each head in Model
            ]
        else:
            # only one head
            next_action = self.env.getBestAction(
                next_output.data, _state_list)

        if Config.double_q:
            # get next outputs, target
            next_output = self.QFunc(self.target_q_func, _next_x)
        return cur_output, next_output, next_action

    def grad(self, _cur_output, _next_output, _next_action,
             _batch_tuples, _err_count=None, _k=None):
        # alloc
        if Config.gpu:
            _cur_output.grad = cupy.zeros_like(_cur_output.data)
        else:
            _cur_output.grad = np.zeros_like(_cur_output.data)

        # compute grad from each tuples
        for i in range(len(_batch_tuples)):
            # if use bootstrap and masked
            if Config.bootstrap and not _batch_tuples[i].mask[_k]:
                continue

            cur_action_value = \
                _cur_output.data[i][_batch_tuples[i].action].tolist()
            reward = _batch_tuples[i].reward
            target_value = reward
            # if not empty position, not terminal state
            if _batch_tuples[i].next_state.in_game:
                next_action_value = \
                    _next_output.data[i][_next_action[i]].tolist()
                target_value += Config.gamma * next_action_value
            loss = cur_action_value - target_value
            _cur_output.grad[i][_batch_tuples[i].action] = 2 * loss

            if Config.prioritized_replay:
                # count err
                if cur_action_value:
                    _batch_tuples[i].err += abs(loss / cur_action_value)
                    if _err_count:
                        _err_count[i] += 1

    def gradWeight(self, _cur_output, _weights):
        # multiply weights with grad
        if Config.gpu:
            _cur_output.grad = cupy.multiply(
                _cur_output.grad, _weights)
        else:
            _cur_output.grad = np.multiply(
                _cur_output.grad, _weights)

    def gradClip(self, _cur_output, _value=1):
        # clip grads
        if Config.gpu:
            _cur_output.grad = cupy.clip(
                _cur_output.grad, -_value, _value)
        else:
            _cur_output.grad = np.clip(
                _cur_output.grad, -_value, _value)

    def train(self):
        # clear grads
        self.q_func.zerograds()

        # pull tuples from memory pool
        batch_tuples = self.replay.pull(Config.batch_size)
        if not len(batch_tuples):
            return

        cur_x, next_x = self.getInputs(batch_tuples)

        # if bootstrap, they are all list for heads
        cur_output, next_output, next_action = self.forward(
            cur_x, next_x, [t.next_state for t in batch_tuples])

        if Config.prioritized_replay:
            # clear err of tuples
            for t in batch_tuples:
                t.err = 0.
            # if prioritized_replay, then we need bias weights
                weights = self.getWeights(batch_tuples)

        if Config.bootstrap:
            if Config.prioritized_replay:
                # store err count
                err_count = [0.] * len(batch_tuples)
            else:
                err_count = None
            # compute grad for each head
            for k in range(Config.K):
                self.grad(cur_output[k], next_output[k], next_action[k],
                          batch_tuples, err_count, k)
                if Config.prioritized_replay:
                    self.gradWeight(cur_output[k], weights)
                if Config.grad_clip:
                    self.gradClip(cur_output[k], Config.grad_clip)
                # backward
                cur_output[k].backward()

            # adjust grads of shared
            for param in self.q_func.shared.params():
                param.grad /= Config.K

            if Config.prioritized_replay:
                # avg err
                for i in range(len(batch_tuples)):
                    if err_count[i] > 0:
                        batch_tuples[i].err /= err_count[i]
        else:
            self.grad(cur_output, next_output, next_action,
                      batch_tuples)
            if Config.prioritized_replay:
                self.gradWeight(cur_output, weights)
            if Config.grad_clip:
                self.gradClip(cur_output, Config.grad_clip)
            # backward
            cur_output.backward()

        # update params
        self.optimizer.update()

        if Config.prioritized_replay:
            self.replay.merge(Config.alpha)
        else:
            self.replay.merge()

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
            if Config.bootstrap:
                output = output[self.use_head]
            return self.env.getBestAction(output.data, [_state])[0]

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
