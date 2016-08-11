from AACAgent import AACAgent
import random
import tensorflow as tf
import numpy as np


class NStepAACAgent(AACAgent):
    """
    Asynchronous Methods for Deep Reinforcement Learning

    Args:
        _model (function): necessary
            return: 1. p func output op, output's dim should be equal with num of actions
                    2. v func output op, output's dim should 1
                    2. vars list
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

    def __init__(self, _model, _env, _is_train=True,
                 _optimizer=None, _global_step=None, _replay=None,
                 _gpu=False, _gamma=0.99, _batch_size=32, _step_len=5,
                 _beta_entropy=0.01, _grad_clip=None, _epoch_show_log=1e3):

        super(NStepAACAgent, self).__init__(
            _model, _env, _is_train,
            _optimizer, _global_step, _replay,
            _gpu, _gamma, _batch_size, _beta_entropy, _grad_clip,
            _epoch_show_log
        )

        self.config.step_len = _step_len

    def step(self):
        """
        Returns:
            still in game or not
        """
        return super(NStepAACAgent, self).nstep(self.p_func)

    def grad(self, _cur_x, _next_output, _batch_tuples, _weights):
        with tf.device(self.config.device):
            # get action data (one hot)
            action_data = self.getActionData(
                self.p_func.get_shape().as_list()[1], _batch_tuples)
            # get target data
            target_data = self.getNStepVTargetData(_next_output, _batch_tuples)
            # get weight data
            weight_data = self.getWeightData(_weights, _batch_tuples)
            # get diff [0], err list [1] and grads [2:]
            ret = self.sess.run(
                [self.err_list_op] + self.grads_op,
                feed_dict={
                    self.x_place: _cur_x,
                    self.action_place: action_data,
                    self.target_place: target_data,
                    self.weight_place: weight_data,
                }
            )
            err_list = ret[0]
            # set grads data
            self.grads_data = ret[1:]

        # return err_list
        return err_list
