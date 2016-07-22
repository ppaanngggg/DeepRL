from ..Model import QModel
from QAgent import QAgent
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


class NStepQAgent(QAgent):
    """
    Asynchronous Methods for Deep Reinforcement Learning

    Args:
        _model (class): necessary, model to create q func,
                        output's dim should be equal with num of actions
        _env (Env): necessary, env to learn, should be rewritten from Env
        _is_train (bool): default True
        _optimizer (chainer.optimizers): not necessary, if not then func won't be updated
        _replay (Replay): necessary for training
        _gpu (bool): whether to use gpu
        _gamma (float): reward decay
        _batch_size (int): how much tuples to pull from replay
        _step_len (int): how much step to do in agent.step()
        _epsilon (float): init epsilon, p for choosing randomly
        _epsilon_decay (float): epsilon *= epsilon_decay
        _epsilon_underline (float): epsilon = max(epsilon_underline, epsilon)
        _grad_clip (float): clip grad, 0 is no clip
    """

    def __init__(self, _model, _env, _is_train=True,
                 _optimizer=None, _replay=None,
                 _gpu=False, _gamma=0.99, _batch_size=32, _step_len=5,
                 _epsilon=0.5, _epsilon_decay=0.995, _epsilon_underline=0.01,
                 _grad_clip=1.):

        super(NStepQAgent, self).__init__(
            _model, _env, _is_train, _optimizer, _replay,
            _gpu, _gamma, _batch_size,
            _epsilon, _epsilon_decay, _epsilon_underline,
            _grad_clip)

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
                action = self.chooseAction(self.q_func, cur_state)
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
            return super(NStepQAgent, self).step()

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
            target_value = 0.
            # if not empty position, not terminal state
            if _batch_tuples[i].next_state.in_game:
                next_action_value = \
                    _next_output.data[i][_next_action[i]].tolist()
                target_value = next_action_value
            for r in reversed(reward):
                target_value = r + self.config.gamma * target_value
            loss = cur_action_value - target_value
            _cur_output.grad[i][_batch_tuples[i].action] = 2 * loss
            err_list.append(abs(loss))
        return err_list
