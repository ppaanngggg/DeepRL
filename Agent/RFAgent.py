from ..Model.ActorCriticModel import Actor, Critic
from Agent import Agent
import random
from chainer import serializers, Variable
import chainer.functions as F
import numpy as np

import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


class RFAgent(Agent):

    def __init__(self, _shared, _actor, _critic, _env, _is_train=True,
                 _actor_optimizer=None, _critic_optimizer=None, _replay=None,
                 _gpu=False, _gamma=0.99, _batch_size=32,
                 _epsilon=0.5, _epsilon_decay=0.995, _epsilon_underline=0.01,
                 _grad_clip=1.):
        """
        Args:
            _shared (class):
            _actor (class):
            _critic (class):
        """

        super(RFAgent, self).__init__()

        self.is_train = _is_train

        self.p_func = Actor(_shared(), _actor())
        self.v_func = Critic(_shared(), _critic())
        self.env = _env
        if self.is_train:
            self.target_v_func = Critic(_shared(), _critic())
            self.target_v_func.copyparams(self.v_func)

            self.p_opt = _actor_optimizer
            self.v_opt = _critic_optimizer
            self.p_opt.setup(self.p_func)
            self.v_opt.setup(self.v_func)
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
        return super(RFAgent, self).step(self.p_func)

    def forward(self, _cur_x, _next_x):
        # get cur outputs
        cur_output = self.func(self.v_func, _cur_x, True)
        # get cur softmax of actor
        cur_softmax = F.softmax(self.func(self.p_func, _cur_x, True))
        # get next outputs, target
        next_output = self.func(self.target_v_func, _next_x, False)
        return cur_output, cur_softmax, next_output

    def grad(self, _cur_output, _cur_softmax, _next_output, _batch_tuples):
        # alloc
        if self.config.gpu:
            _cur_output.grad = cupy.zeros_like(_cur_output.data)
        else:
            _cur_output.grad = np.zeros_like(_cur_output.data)

        cur_action = np.zeros_like(_cur_softmax.data)
        for i in range(len(_batch_tuples)):
            cur_action[i][_batch_tuples[i].action] = 1
        cross_entropy = F.batch_matmul(_cur_softmax, Variable(cur_action),
                                       transa=True)
        cross_entropy = -F.log(cross_entropy)
        # compute grad from each tuples
        err_list = []
        for i in range(len(_batch_tuples)):
            cur_value = _cur_output.data[i][0].tolist()
            reward = _batch_tuples[i].reward
            target_value = reward
            # if not empty position, not terminal state
            if _batch_tuples[i].next_state.in_game:
                next_value = _next_output.data[i][0].tolist()
                target_value += self.config.gamma * next_value
            loss = cur_value - target_value
            cross_entropy.data[i] *= cur_value
            _cur_output.grad[i][0] = 2 * loss
            err_list.append(abs(loss))

        cross_entropy.data /= max(cross_entropy.data.max(),
                                  -cross_entropy.data.min())
        cross_entropy.grad = np.copy(cross_entropy.data)
        for s, g, a in zip(_cur_softmax.data, cross_entropy.grad, _batch_tuples):
            print s, g, a.action
        return err_list, cross_entropy

    def doTrain(self, _batch_tuples, _weights):
        # get inputs from batch
        cur_x = self.getCurInputs(_batch_tuples)
        next_x = self.getNextInputs(_batch_tuples)
        # compute forward
        cur_output, cur_softmax, next_output = self.forward(cur_x, next_x)
        # fill grad
        err_list, cross_entropy = self.grad(
            cur_output, cur_softmax, next_output, _batch_tuples)
        if _weights is not None:
            self.gradWeight(cur_output, _weights)
        if self.config.grad_clip:
            self.gradClip(cur_output, self.config.grad_clip)
        # backward
        cur_output.backward()
        cross_entropy.backward()

        return err_list
