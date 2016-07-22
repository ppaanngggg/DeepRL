from ..Model.ACModel import Actor, Critic
from AACAgent import AACAgent
import random
from chainer import serializers, Variable
import chainer.functions as F
import numpy as np

import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


class NStepAACAgent(AACAgent):
    """
    Asynchronous Methods for Deep Reinforcement Learning

    Args:
        _shared (class): necessary, shared part of func
        _actor (class): necessary, head part of p func,
                        output's dim should be equal with num of actions
        _critic (class): necessary, head part of v func,
                        output's dim should 1
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
        _step_len (int): how much step to do in agent.step()
        _grad_clip (float): clip grad, 0 is no clip
    """

    def __init__(self, _shared, _actor, _critic, _env, _is_train=True,
                 _actor_optimizer=None, _critic_optimizer=None, _replay=None,
                 _gpu=False, _gamma=0.99, _batch_size=32, _step_len=5,
                 _grad_clip=1.):
        """
        Args:
            _shared (class):
            _actor (class):
            _critic (class):
        """

        super(NStepAACAgent, self).__init__(
            _shared, _actor, _critic, _env, _is_train,
            _actor_optimizer, _critic_optimizer, _replay,
            _gpu, _gamma, _batch_size, _grad_clip
        )

        self.config.step_len = _step_len

    def step(self):
        """
        Returns:
            still in game or not
        """
        if self.is_train:
            if not self.env.in_game:
                return False

            state_list = []
            action_list = []
            reward_list = []

            cur_state = self.env.getState()
            for _ in range(self.config.step_len):
                state_list.append(cur_state)
                action = self.chooseAction(self.p_func, cur_state)
                action_list.append(action)
                reward = self.env.doAction(action)
                reward_list.append(reward)
                logger.info('Action: ' + str(action) +
                            '; Reward: %.3f' % (reward))
                next_state = self.env.getState()
                if not self.env.in_game:
                    break
                cur_state = next_state

            for i in range(len(state_list)):
                self.replay.push(state_list[i], action_list[i],
                                 reward_list[i:], next_state)
            return self.env.in_game
        else:
            return super(NStepAACAgent, self).step()

    def grad(self, _cur_output, _cur_softmax, _next_output, _batch_tuples):
        # alloc
        if self.config.gpu:
            _cur_output.grad = cupy.zeros_like(_cur_output.data)
        else:
            _cur_output.grad = np.zeros_like(_cur_output.data)

        cur_action = np.zeros_like(_cur_softmax.data)
        for i in range(len(_batch_tuples)):
            cur_action[i][_batch_tuples[i].action] = 1
        cross_entropy = F.batch_matmul(
            _cur_softmax, Variable(cur_action), transa=True)
        cross_entropy = -F.log(cross_entropy)
        # compute grad from each tuples
        err_list = []
        for i in range(len(_batch_tuples)):
            cur_value = _cur_output.data[i][0].tolist()
            reward = _batch_tuples[i].reward
            target_value = 0.
            # if not empty position, not terminal state
            if _batch_tuples[i].next_state.in_game:
                next_value = _next_output.data[i][0].tolist()
                target_value = next_value
            for r in reversed(reward):
                target_value = r + self.config.gamma * target_value
            loss = cur_value - target_value
            cross_entropy.data[i] *= -loss
            _cur_output.grad[i][0] = 2 * loss
            err_list.append(abs(loss))

        cross_entropy.grad = np.copy(cross_entropy.data)
        return err_list, cross_entropy
