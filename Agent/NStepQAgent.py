from QAgent import QAgent
import random
import tensorflow as tf
import numpy as np

import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


class NStepQAgent(QAgent):
    """
    Asynchronous Methods for Deep Reinforcement Learning

    Args:
        _model (function): necessary, model to create q func,
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
            # buffer
            state_list = []
            action_list = []
            reward_list = []
            # get cur state
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

    def grad(self, _cur_x, _next_output, _next_action, _batch_tuples, _weights):
        with tf.device(self.config.device):
            # get action data (one hot)
            action_data = np.zeros_like(_next_output)
            for i in range(len(_batch_tuples)):
                action_data[i, _batch_tuples[i].action] = 1.
            # get target data
            target_data = np.zeros((len(_batch_tuples)), np.float32)
            for i in range(len(_batch_tuples)):
                reward = _batch_tuples[i].reward
                target_value = 0.
                # if not empty position, not terminal state
                if _batch_tuples[i].next_state.in_game:
                    target_value += _next_output[i][_next_action[i]].tolist()
                for r in reversed(reward):
                    target_value = r + self.config.gamma * target_value
                target_data[i] = target_value
            # get weight data
            if _weights is not None:
                weigth_data = _weights
            else:
                weigth_data = np.ones((len(_batch_tuples)), np.float32)

            # get err list [0] and grads [1:]
            ret = self.sess.run(
                [self.err_list_op] + self.grads_op,
                feed_dict={
                    self.x_place: _cur_x,
                    self.action_place: action_data,
                    self.target_place: target_data,
                    self.weight_place: weigth_data,
                }
            )
            # set grads data
            self.q_grads_data = ret[1:]
        # return err_list
        return ret[0]
