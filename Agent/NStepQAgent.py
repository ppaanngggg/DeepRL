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
        _model (function): necessary,
            return: 1. q func output op, output's dim should be equal with num of actions
                    2. vars list
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
        return super(NStepQAgent, self).nstep(self.q_func)

    def grad(self, _cur_x, _next_output, _next_action, _batch_tuples, _weights):
        with tf.device(self.config.device):
            # get action data (one hot)
            action_data = self.getActionData(
                self.q_func.get_shape().as_list()[1], _batch_tuples)
            # get target data
            target_data = self.getNStepQTargetData(
                _next_output, _next_action, _batch_tuples)
            # get weight data
            weight_data = self.getWeightData(_weights, _batch_tuples)

            # get err list [0] and grads [1:]
            ret = self.sess.run(
                [self.err_list_op] + self.grads_op,
                feed_dict={
                    self.x_place: _cur_x,
                    self.action_place: action_data,
                    self.target_place: target_data,
                    self.weight_place: weight_data,
                }
            )
            # set grads data
            self.grads_data = ret[1:]
        # return err_list
        return ret[0]
