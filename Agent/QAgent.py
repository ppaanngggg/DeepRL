from ..Model.QModel import QModel
from Agent import Agent
import random
from chainer import serializers
import numpy as np

import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


class QAgent(Agent):

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
        self.env = _env
        if self.is_train:
            self.target_q_func = QModel(_model())
            self.target_q_func.copyparams(self.q_func)

            self.optimizer = _optimizer
            self.optimizer.setup(self.q_func)
            self.replay = _replay

        self.gpu = _gpu
        self.gamma = _gamma
        self.batch_size = 32
        self.epsilon = _epsilon
        self.epsilon_decay = _epsilon_decay
        self.epsilon_underline = _epsilon_underline
        self.grad_clip = _grad_clip

    def step(self):
        """
        Returns:
            still in game or not
        """
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
            self.replay.push(cur_state, action, reward, next_state)

        return self.env.in_game

    def forward(self, _cur_x, _next_x, _state_list):
        # get cur outputs
        cur_output = self.QFunc(self.q_func, _cur_x, True)
        # get next outputs, NOT target
        next_output = self.QFunc(self.q_func, _next_x, False)

        # only one head
        next_action = self.env.getBestAction(
            next_output.data, _state_list)

        # get next outputs, target
        next_output = self.QFunc(self.target_q_func, _next_x, False)
        return cur_output, next_output, next_action

    def grad(self, _cur_output, _next_output, _next_action, _batch_tuples):
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
                next_action_value = \
                    _next_output.data[i][_next_action[i]].tolist()
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
        cur_output, next_output, next_action = self.forward(
            cur_x, next_x, [t.next_state for t in batch_tuples])
        # fill grad
        err_list = self.grad(cur_output, next_output,
                             next_action, batch_tuples)
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
                return self.env.getBestAction(output.data, [_state])[0]
        else:
            x_data = self.env.getX(_state)
            output = self.QFunc(_model, x_data, False)
            return self.env.getBestAction(output.data, [_state])[0]
