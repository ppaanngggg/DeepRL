from ..Model import QModel
from Agent import Agent
import random
from chainer import cuda
try:
    import cupy
except:
    pass
import numpy as np

import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


class QAgent(Agent):
    """
        Human-level control through deep reinforcement learning
    """

    def __init__(self, _model, _env, _is_train=True,
                 _optimizer=None, _replay=None,
                 _gpu=False, _gamma=0.99, _batch_size=32,
                 _epsilon=0.5, _epsilon_decay=0.995, _epsilon_underline=0.01,
                 _grad_clip=1.):
        """
        Args:
            _model (class): model
        """

        super(QAgent, self).__init__()

        self.is_train = _is_train

        self.q_func = QModel(_model())
        if _gpu:
            self.q_func.to_gpu()
        self.env = _env
        if self.is_train:
            self.target_q_func = QModel(_model())
            if _gpu:
                self.target_q_func.to_gpu()
            self.target_q_func.copyparams(self.q_func)

            if _optimizer:
                self.q_opt = _optimizer
                self.q_opt.setup(self.q_func)
            self.replay = _replay

        self.config.gpu = _gpu
        self.config.gamma = _gamma
        self.config.batch_size = _batch_size
        self.config.epsilon = _epsilon
        self.config.epsilon_decay = _epsilon_decay
        self.config.epsilon_underline = _epsilon_underline
        self.config.grad_clip = _grad_clip

    def step(self):
        """
        Returns:
            still in game or not
        """
        return super(QAgent, self).step(self.q_func)

    def forward(self, _cur_x, _next_x, _state_list):
        # get cur outputs
        cur_output = self.func(self.q_func, _cur_x, True)
        # get next outputs, NOT target
        next_output = self.func(self.q_func, _next_x, False)

        # only one head
        next_action = self.env.getBestAction(next_output.data, _state_list)

        # get next outputs, target
        next_output = self.func(self.target_q_func, _next_x, False)
        return cur_output, next_output, next_action

    def grad(self, _cur_output, _next_output, _next_action, _batch_tuples):
        # alloc
        if self.config.gpu:
            _cur_output.grad = cupy.zeros_like(_cur_output.data)
        else:
            _cur_output.grad = np.zeros_like(_cur_output.data)

        # compute grad from each tuples
        err_list = []
        for i in range(len(_batch_tuples)):
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
            err_list.append(abs(loss))
        return err_list

    def doTrain(self, _batch_tuples, _weights):
        # get inputs from batch
        cur_x = self.getCurInputs(_batch_tuples)
        next_x = self.getNextInputs(_batch_tuples)
        # compute forward
        cur_output, next_output, next_action = self.forward(
            cur_x, next_x, [t.next_state for t in _batch_tuples])
        # fill grad
        err_list = self.grad(cur_output, next_output,
                             next_action, _batch_tuples)
        if _weights is not None:
            if self.config.gpu:
                _weights = cuda.to_gpu(_weights)
            self.gradWeight(cur_output, _weights)
        if self.config.grad_clip:
            self.gradClip(cur_output, self.config.grad_clip)
        # backward
        cur_output.backward()
        return err_list
