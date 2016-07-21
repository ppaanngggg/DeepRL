from ..Model import BootQModel
from Agent import Agent
import random
import numpy as np

import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


class BootQAgent(Agent):
    """
    Deep Exploration via Bootstrapped DQN

    Args:
        _shard (class): necessary, shared part of q func
        _head (class): necessary, head part of q func
        _env (Env): necessary, env to learn, should be rewritten from Env
        _is_train (bool): default True
        _optimizer (chainer.optimizers): not necessary, if not then func won't be updated
        _replay (Replay): necessary for training
        _K (int): how many heads to use
        _mask_p (float): p to be passed when train for each head
        _gpu (bool): whether to use gpu
        _gamma (float): reward decay
        _batch_size (int): how much tuples to pull from replay
        _epsilon (float): init epsilon, p for choosing randomly
        _epsilon_decay (float): epsilon *= epsilon_decay
        _epsilon_underline (float): epsilon = max(epsilon_underline, epsilon)
        _grad_clip (float): clip grad, 0 is no clip
    """

    def __init__(self, _shared, _head, _env, _is_train=True,
                 _optimizer=None, _replay=None,
                 _K=10, _mask_p=0.5,
                 _gpu=False, _gamma=0.99, _batch_size=32,
                 _epsilon=0.5, _epsilon_decay=0.995, _epsilon_underline=0.01,
                 _grad_clip=1.):

        super(BootQAgent, self).__init__()

        self.is_train = _is_train

        self.q_func = BootQModel(_shared, _head, _K)
        self.env = _env
        if self.is_train:
            self.target_q_func = BootQModel(_shared, _head, _K)
            self.target_q_func.copyparams(self.q_func)

            if _optimizer:
                self.q_opt = _optimizer
                self.q_opt.setup(self.q_func)
                self.replay = _replay

        self.config.K = _K
        self.config.mask_p = _mask_p
        self.config.gpu = _gpu
        self.config.gamma = _gamma
        self.config.batch_size = _batch_size
        self.config.epsilon = _epsilon
        self.config.epsilon_decay = _epsilon_decay
        self.config.epsilon_underline = _epsilon_underline
        self.config.grad_clip = _grad_clip

    def startNewGame(self):
        super(BootQAgent, self).startNewGame()
        self.use_head = random.randint(0, self.config.K - 1)
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
                np.random.binomial(1, self.config.mask_p,
                                   (self.config.K)).tolist()
            )

        return self.env.in_game

    def forward(self, _cur_x, _next_x, _state_list):
        # get cur outputs
        cur_output = self.func(self.q_func, _cur_x, True)
        # get next outputs, NOT target
        next_output = self.func(self.q_func, _next_x, False)

        # choose next action for each output
        next_action = [
            self.env.getBestAction(
                o.data,
                _state_list
            ) for o in next_output  # for each head in Model
        ]

        # get next outputs, target
        next_output = self.func(self.target_q_func, _next_x, False)
        return cur_output, next_output, next_action

    def grad(self, _cur_output, _next_output, _next_action,
             _batch_tuples, _err_list, _err_count, _k):
        # alloc
        if self.config.gpu:
            _cur_output.grad = cupy.zeros_like(_cur_output.data)
        else:
            _cur_output.grad = np.zeros_like(_cur_output.data)

        # compute grad from each tuples
        for i in range(len(_batch_tuples)):
            # if use bootstrap and masked
            if not _batch_tuples[i].mask[_k]:
                continue

            cur_action_value = \
                _cur_output.data[i][_batch_tuples[i].action].tolist()
            reward = _batch_tuples[i].reward
            target_value = reward
            # if not empty position, not terminal state
            if _batch_tuples[i].next_state.in_game:
                next_action_value = \
                    _next_output.data[i][_next_action[i]].tolist()
                target_value += self.config.gamma * next_action_value
            loss = cur_action_value - target_value
            _cur_output.grad[i][_batch_tuples[i].action] = 2 * loss

            _err_list[i] += abs(loss)
            _err_count[i] += 1

    def doTrain(self, _batch_tuples, _weights):
        cur_x = self.getCurInputs(_batch_tuples)
        next_x = self.getNextInputs(_batch_tuples)
        # if bootstrap, they are all list for heads
        cur_output, next_output, next_action = self.forward(
            cur_x, next_x, [t.next_state for t in _batch_tuples])
        # compute grad for each head
        err_list = [0.] * len(_batch_tuples)
        err_count = [0.] * len(_batch_tuples)
        for k in range(self.config.K):
            self.grad(cur_output[k], next_output[k], next_action[k],
                      _batch_tuples, err_list, err_count, k)
            if _weights is not None:
                self.gradWeight(cur_output[k], _weights)
            if self.config.grad_clip:
                self.gradClip(cur_output[k], self.config.grad_clip)
            # backward
            cur_output[k].backward()

        # adjust grads of shared
        for param in self.q_func.shared.params():
            param.grad /= self.config.K

        # avg err
        for i in range(len(err_list)):
            if err_count[i] > 0:
                err_list[i] /= err_count[i]
            else:
                err_list[i] = None
        return err_list

    def chooseAction(self, _model, _state):
        if self.is_train:
            # update epsilon
            self.updateEpsilon()
            random_value = random.random()
            if random_value < self.config.epsilon:
                # randomly choose
                return self.env.getRandomAction(_state)
            else:
                # use model to choose
                x_data = self.env.getX(_state)
                output = self.func(_model, x_data, False)
                output = output[self.use_head]
                return self.env.getBestAction(output.data, [_state])[0]
        else:
            x_data = self.env.getX(_state)
            output = self.func(_model, x_data, False)
            action_dict = {}
            for o in output:
                action = self.env.getBestAction(o.data, [_state])[0]
                if action not in action_dict.keys():
                    action_dict[action] = 1
                else:
                    action_dict[action] += 1
            logger.info(str(action_dict))
            max_k = -1
            max_v = 0
            for k, v in zip(action_dict.keys(), action_dict.values()):
                if v > max_v:
                    max_k = k
                    max_v = v
            return max_k
