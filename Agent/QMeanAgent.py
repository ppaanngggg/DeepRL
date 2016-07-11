from ..Model.QModel import QModel
from Agent import QAgent
import random
from chainer import serializers
import numpy as np

import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


class QMeanAgent(QAgent):

    def __init__(self, _model, _env, _is_train=True,
                 _optimizer=None, _replay=None,
                 _gpu=False, _gamma=0.99, _batch_size=32,
                 _epsilon=0.5, _epsilon_decay=0.995, _epsilon_underline=0.01,
                 _grad_clip=1.):
        """
        Args:
            _model (class): model
        """

        super(QMeanAgent, self).__init__(
            _model, _env, _is_train, _optimizer,
            _replay, _gpu, _gamma, _batch_size,
            _epsilon, _epsilon_decay, _epsilon_underline,
            _grad_clip
        )

    def forward(self, _cur_x, _next_x, _state_list):
        # get cur outputs
        cur_output = self.func(self.q_func, _cur_x, True)
        # get next outputs, target
        next_output = self.func(self.target_q_func, _next_x, False)
        return cur_output, next_output

    def grad(self, _cur_output, _next_output, _batch_tuples):
        # alloc
        if self.gpu:
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
                next_action_value = _next_output.data[i].mean()
                target_value += self.gamma * next_action_value
            loss = cur_action_value - target_value
            _cur_output.grad[i][_batch_tuples[i].action] = 2 * loss
            err_list.append(abs(loss))
        return err_list

    def train(self):
        # clear grads
        self.q_func.zerograds()

        # pull tuples from memory pool
        batch_tuples, weights = self.replay.pull(self.batch_size)
        if not len(batch_tuples):
            return

        # get inputs from batch
        cur_x, next_x = self.getInputs(batch_tuples)
        # compute forward
        cur_output, next_output = self.forward(
            cur_x, next_x, [t.next_state for t in batch_tuples])
        # fill grad
        err_list = self.grad(cur_output, next_output, batch_tuples)
        if weights is not None:
            self.gradWeight(cur_output, weights)
        if self.grad_clip:
            self.gradClip(cur_output, self.grad_clip)
        # backward
        cur_output.backward()
        # update params
        self.optimizer.update()

        # merget tmp replay into pool
        self.replay.setErr(batch_tuples, err_list)
        self.replay.merge()
