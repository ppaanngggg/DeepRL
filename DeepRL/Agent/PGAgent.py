from Agent import Agent
import random
import tensorflow as tf
import numpy as np


class PGAgent(Agent):

    def __init__(self, _model, _env, _is_train=True,
                 _optimizer=None, _global_step=None, _replay=None,
                 _gpu=False, _gamma=0.99,
                 _batch_size=32, _beta_entropy=0.01,
                 _grad_clip=None, _epoch_show_log=1e3):

        super(PGAgent, self).__init__(_is_train, _gpu)

        # set config
        self.config.gpu = _gpu
        self.config.gamma = _gamma
        self.config.batch_size = _batch_size
        self.config.beta_entropy = _beta_entropy
        self.config.grad_clip = _grad_clip
        self.config.epoch_show_log = _epoch_show_log

        # set env
        self.env = _env

        with tf.device(self.config.device):
            # create p func
            self.p_func, self.vars = _model(self.x_place)

            if self.is_train:
                # place for action, value
                self.value_place = tf.placeholder(tf.float32)
                self.action_place = tf.placeholder(tf.float32)
                # get entropy
                entropy = tf.reduce_sum(
                    self.p_func * tf.log(self.p_func + 1e-10))
                # get loss
                loss = -tf.reduce_sum(
                    tf.log(
                        tf.reduce_sum(self.p_func * self.action_place, 1)
                        + 1e-10) * self.value_place) + \
                    self.config.beta_entropy * entropy

                # compute grads of vars
                self.grads_op = tf.gradients(loss, self.vars)

                if _optimizer:
                    self.createOpt(_optimizer, _global_step)

                self.replay = _replay

            # init all vars
            self.sess.run(tf.initialize_all_variables())

    def step(self):
        return super(PGAgent, self).stepUntilEnd(self.p_func)

    def grad(self, _cur_x, _batch_tuples):
        with tf.device(self.config.device):
            # get action data (one hot)
            action_data = self.getActionData(
                self.p_func.get_shape().as_list()[1], _batch_tuples)
            # get value data
            value_data = self.getNStepVTargetData(None, _batch_tuples)
            if value_data.std() == 0:
                value_data = np.zero_like(value_data)
            else:
                value_data = (value_data - value_data.mean()) / \
                    value_data.std()
            self.grads_data = self.sess.run(
                self.grads_op,
                feed_dict={
                    self.x_place: _cur_x,
                    self.action_place: action_data,
                    self.value_place: value_data,
                }
            )

    def doTrain(self, _batch_tuples, _weights):
        # get inputs from batch
        cur_x = self.getCurInputs(_batch_tuples)
        # fill grad
        self.grad(cur_x, _batch_tuples)

        return np.ones([len(_batch_tuples)], np.float32)

    def chooseAction(self, _model, _state):
        return self.chooseSoftAction(_model, _state)
