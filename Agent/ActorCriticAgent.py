from ..Model.ActorCriticModel import Actor, Critic
from QAgent import QAgent
import random
from chainer import serializers, Variable
import chainer.functions as F
import numpy as np

import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


class ActorCriticAgent(QAgent):

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

        self.is_train = _is_train

        self.p_func = Actor(_shared(), _actor())
        self.q_func = Critic(_shared(), _critic())
        self.env = _env
        if self.is_train:
            self.target_q_func = Critic(_shared(), _critic())
            self.target_q_func.copyparams(self.q_func)

            self.actor_optimizer = _actor_optimizer
            self.critic_optimizer = _critic_optimizer
            self.actor_optimizer.setup(self.p_func)
            self.critic_optimizer.setup(self.q_func)
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
        action = self.chooseAction(self.p_func, cur_state)
        # do action and get reward
        reward = self.env.doAction(action)

        logger.info('Action: ' + str(action) + '; Reward: %.3f' % (reward))

        if self.is_train:
            # get new state
            next_state = self.env.getState()
            # store replay_tuple into memory pool
            self.replay.push(cur_state, action, reward, next_state)

        return self.env.in_game

    def forward(self, _cur_x, _next_x):
        # get cur outputs
        cur_output = self.func(self.q_func, _cur_x, True)
        # get cur softmax of actor
        cur_softmax = F.softmax(self.func(self.p_func, _cur_x, True))
        cur_softmax.data[cur_softmax.data < 0.01] = 0.01
        # get next outputs, target
        next_output = self.func(self.target_q_func, _next_x, False)
        return cur_output, cur_softmax, next_output

    def grad(self, _cur_output, _cur_softmax, _next_output, _batch_tuples):
        # alloc
        if self.gpu:
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
                target_value += self.gamma * next_value
            loss = cur_value - target_value
            cross_entropy.data[i] *= target_value
            _cur_output.grad[i][0] = 2 * loss
            err_list.append(abs(loss))

        cross_entropy.grad = np.copy(cross_entropy.data)
        # for s, g, a in zip(_cur_softmax.data, cross_entropy.grad, _batch_tuples):
        #     print s, g, a.action
        # raw_input()
        return err_list, cross_entropy

    def train(self):
        # clear grads
        self.p_func.zerograds()
        self.q_func.zerograds()

        # pull tuples from memory pool
        batch_tuples, weights = self.replay.pull(self.batch_size)
        if not len(batch_tuples):
            return

        # get inputs from batch
        cur_x, next_x = self.getInputs(batch_tuples)
        # compute forward
        cur_output, cur_softmax, next_output = self.forward(cur_x, next_x)
        # fill grad
        err_list, cross_entropy = self.grad(
            cur_output, cur_softmax, next_output, batch_tuples)
        self.replay.setErr(batch_tuples, err_list)
        if weights is not None:
            self.gradWeight(cur_output, weights)
        if self.grad_clip:
            self.gradClip(cur_output, self.grad_clip)
        # backward
        cur_output.backward()
        cross_entropy.backward()
        # update params
        self.actor_optimizer.update()
        self.critic_optimizer.update()

        # merget tmp replay into pool
        self.replay.merge()

    def save(self, _epoch, _step):
        filename = './models/epoch_' + str(_epoch) + '_step_' + str(_step)
        logger.info(filename)
        serializers.save_npz(filename, self.p_func)

    def load(self, filename):
        logger.info(filename)
        serializers.load_npz(filename, self.p_func)
