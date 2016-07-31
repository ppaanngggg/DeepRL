from AACAgent import AACAgent
import random
import tensorflow as tf
import numpy as np

import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


class NStepAACAgent(AACAgent):
    """
    Asynchronous Methods for Deep Reinforcement Learning

    Args:
        _actor (function): necessary, head part of p func,
                        output's dim should be equal with num of actions
        _critic (function): necessary, head part of v func,
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

    def __init__(self, _actor, _critic, _env, _is_train=True,
                 _actor_optimizer=None, _critic_optimizer=None, _replay=None,
                 _gpu=False, _gamma=0.99, _batch_size=32, _step_len=5,
                 _beta_entropy=0.01, _grad_clip=1.):

        super(NStepAACAgent, self).__init__(
            _actor, _critic, _env, _is_train,
            _actor_optimizer, _critic_optimizer, _replay,
            _gpu, _gamma, _batch_size, _beta_entropy, _grad_clip
        )

        self.config.step_len = _step_len

    def step(self):
        """
        Returns:
            still in game or not
        """
        return super(NStepAACAgent, self).nstep(self.softmax_op)

    def grad(self, _cur_x, _next_output, _batch_tuples, _weights):
        with tf.device(self.config.device):
            # critic part
            # get target data
            target_data = self.getNStepVTargetData(_next_output, _batch_tuples)
            # get weight data
            weight_data = self.getWeightData(_weights, _batch_tuples)
            # get diff [0], err list [1] and grads [2:]
            ret = self.sess.run(
                [self.diff_op, self.err_list_op] + self.critic_grads_op,
                feed_dict={
                    self.x_place: _cur_x,
                    self.target_place: target_data,
                    self.weight_place: weight_data,
                }
            )
            diff_data = ret[0]
            err_list = ret[1]
            # set v grads data
            self.v_grads_data = ret[2:]

            # actor part
            # get action data (one hot)
            action_data = self.getActionData(
                self.softmax_op.get_shape().as_list()[1], _batch_tuples)
            # set p grad data
            self.p_grads_data = self.sess.run(
                self.actor_grads_op,
                feed_dict={
                    self.x_place: _cur_x,
                    self.action_place: action_data,
                    self.diff_place: diff_data,
                }
            )
        # return err_list
        return err_list
