from ..Model.ACModel import Actor, Critic
from Agent import Agent
import random
from chainer import serializers, Variable
import chainer.functions as F
import numpy as np

import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


class QACAgent(Agent):
    """
    Determinisitc Policy Gradient Algorithms

    Args:
        _shared (class): necessary, shared part of func
        _actor (class): necessary, head part of p func,
                        output's dim should be equal with num of actions
        _critic (class): necessary, head part of q func,
                        output's dim should be equal with num of actions
        _env (Env): necessary, env to learn, should be rewritten from Env
        _is_train (bool): default True
        _actor_optimizer (chainer.optimizers): not necessary, opter for actor,
                                                if not then func won't be updated
        _critic_optimizer (chainer.optimizers): not necessary, opter for critic,
                                                if not then func won't be updated
        _replay (Replay): necessary for training
        _gpu (bool): whether to use gpu
        _gamma (float): reward decay
        _batch_size (int): how much tuples to pull from replay
        _grad_clip (float): clip grad, 0 is no clip
    """

    def __init__(self, _shared, _actor, _critic, _env, _is_train=True,
                 _actor_optimizer=None, _critic_optimizer=None, _replay=None,
                 _gpu=False, _gamma=0.99, _batch_size=32,
                 _grad_clip=1.):


        super(QACAgent, self).__init__()

        self.is_train = _is_train

        self.p_func = Actor(_shared(), _actor())
        self.q_func = Critic(_shared(), _critic())
        self.env = _env
        if self.is_train:
            if _actor_optimizer:
                self.p_opt = _actor_optimizer
                self.p_opt.setup(self.p_func)
            if _critic_optimizer:
                self.q_opt = _critic_optimizer
                self.q_opt.setup(self.q_func)

            self.replay = _replay

        self.config.gpu = _gpu
        self.config.gamma = _gamma
        self.config.batch_size = _batch_size
        self.config.grad_clip = _grad_clip

    def step(self):
        """
        Returns:
            still in game or not
        """
        return super(QACAgent, self).step(self.p_func)

    def forward(self, _cur_x, _next_x, _state_list):
        # get cur outputs
        cur_output = self.func(self.q_func, _cur_x, True)
        # get cur softmax of actor
        cur_softmax = F.softmax(self.func(self.p_func, _cur_x, True))
        # get next outputs
        next_output = self.func(self.q_func, _next_x, False)

        tmp_next_output = self.func(self.p_func, _next_x, False)
        next_action = self.env.getBestAction(tmp_next_output.data, _state_list)

        return cur_output, cur_softmax, next_output, next_action

    def grad(self, _cur_output, _cur_softmax,
             _next_output, _next_action, _batch_tuples):
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
            cur_value = _cur_output.data[i][_batch_tuples[i].action].tolist()
            reward = _batch_tuples[i].reward
            target_value = reward
            # if not empty position, not terminal state
            if _batch_tuples[i].next_state.in_game:
                next_value = _next_output.data[i][_next_action[i]].tolist()
                target_value += self.config.gamma * next_value
            loss = cur_value - target_value
            cross_entropy.data[i] *= cur_value
            _cur_output.grad[i][_batch_tuples[i].action] = 2 * loss
            err_list.append(abs(loss))

        cross_entropy.grad = np.copy(cross_entropy.data)
        # for s, g, a in zip(_cur_softmax.data, cross_entropy.grad, _batch_tuples):
        #     print s, g, a.action
        return err_list, cross_entropy

    def doTrain(self, _batch_tuples, _weights):
        # get inputs from batch
        cur_x = self.getCurInputs(_batch_tuples)
        next_x = self.getNextInputs(_batch_tuples)
        # compute forward
        cur_output, cur_softmax, next_output, next_action = self.forward(
            cur_x, next_x, [t.next_state for t in _batch_tuples])
        # fill grad
        err_list, cross_entropy = self.grad(
            cur_output, cur_softmax, next_output, next_action, _batch_tuples)
        if _weights is not None:
            self.gradWeight(cur_output, _weights)
        if self.config.grad_clip:
            self.gradClip(cur_output, self.config.grad_clip)
        # backward
        cur_output.backward()
        cross_entropy.backward()

        return err_list

    def chooseAction(self, _model, _state):
        x_data = self.env.getX(_state)
        output = self.func(_model, x_data, False)
        logger.info(str(F.softmax(output).data))
        if self.is_train:
            return self.env.getSoftAction(output.data, [_state])[0]
        else:
            return self.env.getBestAction(output.data, [_state])[0]
