from ..Model.BootstrappedQModel import BootstrappedQModel
import random
from chainer import serializers, optimizers

import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


class BootstrappedQAgent(object):

    def __init__(self, _shared, _head, _env, _is_train=True,
                 _optimizer=None, _replay=None,
                 _K=10, _mask_p=0.5,
                 _gpu=False, _gamma=0.99, _batch_size=32,
                 _epsilon=0.5, _epsilon_decay=0.995, _epsilon_underline=0.01,
                 _grad_clip=1.):
        """
        Args:
            _shard (class): shared model
            _head (class): head model
        """
        super(BootstrappedQAgent, self).__init__()

        self.is_train = _is_train

        self.q_func = BootstrappedQModel(_shared, _head, _K)
        self.env = _env
        if self.is_train:
            self.target_q_func = BootstrappedQModel(_shared, _head, _K)
            self.target_q_func.copyparams(q_func)

            self.optimizer = _optimizer
            self.optimizer.setup(self.q_func)
            self.replay = _replay

        self.K = _K
        self.mask_p = _mask_p
        self.gpu = _gpu
        self.gamma = _gamma
        self.batch_size = 32
        self.epsilon = _epsilon
        self.epsilon_decay = _epsilon_decay
        self.epsilon_underline = _epsilon_underline
        self.grad_clip = _grad_clip

    def startNewGame(self):
        super(BootstrappedQAgent, self).startNewGame():
        self.use_head = random.randint(0, self.K - 1)
        logger.info('Use head: ' + str(self.use_head))

    def step(self):
        if not self.env.in_game:
            return False

        # get current state
        cur_state = self.env.getState()
        # choose action in step
        action = self.chooseAction(self.q_func, cur_state)
        # do action and get reward
        reward = self.env.doAction(action)

        logger.info('Action: ' + str(action) + '; Reward: %.3f' % (reward))

        if self.is_train:
            # get new state
            next_state = self.env.getState()
            # store replay_tuple into memory pool
            self.replay.push(
                cur_state, action, reward, next_state,
                np.random.binomial(1, self.p, (self.K)).tolist()
            )

        return self.env.in_game

    def forward(self, _cur_x, _next_x, _state_list):
        # get cur outputs
        cur_output = self.QFunc(self.q_func, _cur_x, True)
        if self.double_q:
            # get next outputs, NOT target
            next_output = self.QFunc(self.q_func, _next_x, False)
        else:
            next_output = self.QFunc(self.target_q_func, _next_x, False)

        if self.bootstrap:
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

        if self.double_q:
            # get next outputs, target
            next_output = self.QFunc(self.target_q_func, _next_x, False)
        return cur_output, next_output, next_action

    def grad(self, _cur_output, _next_output, _next_action,
             _batch_tuples, _err_count=None, _k=None):
        # alloc
        if self.gpu:
            _cur_output.grad = cupy.zeros_like(_cur_output.data)
        else:
            _cur_output.grad = np.zeros_like(_cur_output.data)

        # compute grad from each tuples
        for i in range(len(_batch_tuples)):
            # if use bootstrap and masked
            if self.bootstrap and not _batch_tuples[i].mask[_k]:
                continue

            cur_action_value = \
                _cur_output.data[i][_batch_tuples[i].action].tolist()
            reward = _batch_tuples[i].reward
            target_value = reward
            # if not empty position, not terminal state
            if _batch_tuples[i].next_state.in_game:
                next_action_value = \
                    _next_output.data[i][_next_action[i]].tolist()
                target_value += self.gamma * next_action_value
            loss = cur_action_value - target_value
            _cur_output.grad[i][_batch_tuples[i].action] = 2 * loss

            if self.prioritized_replay:
                # count err
                if cur_action_value:
                    _batch_tuples[i].err += abs(loss / cur_action_value)
                    if _err_count:
                        _err_count[i] += 1

    def gradWeight(self, _cur_output, _weights):
        # multiply weights with grad
        if self.gpu:
            _cur_output.grad = cupy.multiply(
                _cur_output.grad, _weights)
        else:
            _cur_output.grad = np.multiply(
                _cur_output.grad, _weights)

    def gradClip(self, _cur_output, _value=1):
        # clip grads
        if self.gpu:
            _cur_output.grad = cupy.clip(
                _cur_output.grad, -_value, _value)
        else:
            _cur_output.grad = np.clip(
                _cur_output.grad, -_value, _value)

    def train(self):
        # clear grads
        self.q_func.zerograds()

        # pull tuples from memory pool
        batch_tuples = self.replay.pull(self.batch_size)
        if not len(batch_tuples):
            return

        cur_x, next_x = self.getInputs(batch_tuples)

        # if bootstrap, they are all list for heads
        cur_output, next_output, next_action = self.forward(
            cur_x, next_x, [t.next_state for t in batch_tuples])

        if self.prioritized_replay:
            # clear err of tuples
            for t in batch_tuples:
                t.err = 0.
            # if prioritized_replay, then we need bias weights
                weights = self.getWeights(batch_tuples)

        if self.bootstrap:
            if self.prioritized_replay:
                # store err count
                err_count = [0.] * len(batch_tuples)
            else:
                err_count = None
            # compute grad for each head
            for k in range(self.K):
                self.grad(cur_output[k], next_output[k], next_action[k],
                          batch_tuples, err_count, k)
                if self.prioritized_replay:
                    self.gradWeight(cur_output[k], weights)
                if self.grad_clip:
                    self.gradClip(cur_output[k], self.grad_clip)
                # backward
                cur_output[k].backward()

            # adjust grads of shared
            for param in self.q_func.shared.params():
                param.grad /= self.K

            if self.prioritized_replay:
                # avg err
                for i in range(len(batch_tuples)):
                    if err_count[i] > 0:
                        batch_tuples[i].err /= err_count[i]
        else:
            self.grad(cur_output, next_output, next_action,
                      batch_tuples)
            if self.prioritized_replay:
                self.gradWeight(cur_output, weights)
            if self.grad_clip:
                self.gradClip(cur_output, self.grad_clip)
            # backward
            cur_output.backward()

        # update params
        self.optimizer.update()

        if self.prioritized_replay:
            self.replay.merge(self.alpha)
        else:
            self.replay.merge()

    def chooseAction(self, _model, _state):
        if self.is_train:
            # update epsilon
            self.epsilon = max(
                self.epsilon_underline,
                self.epsilon * self.epsilon_decay
            )
            random_value = random.random()
            if random_value < self.epsilon:
                # randomly choose
                return self.env.getRandomAction(_state)
            else:
                # use model to choose
                x_data = self.env.getX(_state)
                output = self.QFunc(_model, x_data, False)
                output = output[self.use_head]
                return self.env.getBestAction(output.data, [_state])[0]
        else:
            x_data = self.env.getX(_state)
            output = self.QFunc(_model, x_data, False)
            action_dict = {}
            for o in output:
                action = self.env.getBestAction(o.data, [_state])[0]
                if action not in action_dict.keys():
                    action_dict[action] = 1
                else:
                    action_dict[action] += 1
            max_k = -1
            max_v = 0
            for k, v in zip(action_dict.keys(), action_dict.values()):
                if v > max_v:
                    max_k = k
                    max_v = v
            return max_k

    def updateTargetQFunc(self):
        logger.info('')
        self.target_q_func.copyparams(self.q_func)

    def save(self, _step):
        filename = './models/step_' + str(_step)
        logger.info(filename)
        serializers.save_npz(filename, self.q_func)
