from Agent import Agent
import random
import tensorflow as tf
import numpy as np

import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


class AACAgent(Agent):
    """
    Asynchronous Methods for Deep Reinforcement Learning

    Args:
        _model (function): necessary
            return: 1. p func output op, output's dim should be equal with num of actions
                    2. v func output op, output's dim should 1
                    2. vars list
        _env (Env): necessary, env to learn, should be rewritten from Env
        _is_train (bool): default True
        _optimizer (optimizers): not necessary, if not then func won't be updated
        _replay (Replay): necessary for training
        _gpu (bool): whether to use gpu
        _gamma (float): reward decay
        _batch_size (int): how much tuples to pull from replay
        _beta_entropy (float): beta for entropy
        _grad_clip (float): clip grad, 0 is no clip
    """

    def __init__(self, _model, _env, _is_train=True,
                 _optimizer=None, _replay=None,
                 _gpu=False, _gamma=0.99,
                 _batch_size=32, _beta_entropy=0.01,
                 _grad_clip=1.):

        super(AACAgent, self).__init__(_is_train, _gpu)

        # set env
        self.env = _env

        with tf.device(self.config.device):
            # create p func
            self.p_func, self.v_func, self.vars = _model(self.x_place)

            if self.is_train:
                # place for target, weight
                self.target_place = tf.placeholder(tf.float32)
                self.weight_place = tf.placeholder(tf.float32)
                # get diff of target and v
                diff_op = self.target_place - tf.reshape(self.v_func, [-1])
                # get err of value and target
                self.err_list_op = 0.5 * tf.square(diff_op)
                # place for action, diff
                self.action_place = tf.placeholder(tf.float32)
                # get entropy
                entropy = tf.reduce_sum(
                    self.p_func * tf.log(self.p_func + 1e-10))
                # get loss
                loss = tf.reduce_sum(
                    tf.log(
                        tf.reduce_sum(self.p_func * self.action_place, 1)
                        + 1e-10
                    ) * diff_op
                ) + _beta_entropy * entropy + tf.reduce_mean(self.err_list_op * self.weight_place)
                # compute grads of vars
                self.grads_op = tf.gradients(loss, self.vars)

                if _optimizer:
                    self.createOpt(_optimizer)

                self.replay = _replay

            # init all vars
            self.sess.run(tf.initialize_all_variables())
            # copy params from q func to target
            self.updateTargetFunc()

        self.config.gpu = _gpu
        self.config.gamma = _gamma
        self.config.batch_size = _batch_size
        self.config.beta_entropy = _beta_entropy
        self.config.grad_clip = _grad_clip

    def step(self):
        """
        Returns:
            still in game or not
        """
        return super(AACAgent, self).step(self.p_func)

    def forward(self, _next_x):
        with tf.device(self.config.device):
            # get next outputs, target
            next_output = self.func(self.v_func, _next_x, False)
        return next_output

    def grad(self, _cur_x, _next_output, _batch_tuples, _weights):
        with tf.device(self.config.device):
            # get action data (one hot)
            action_data = self.getActionData(
                self.p_func.get_shape().as_list()[1], _batch_tuples)
            # get target data
            target_data = self.getVTargetData(_next_output, _batch_tuples)
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
            err_list = ret[0]
            # set v grads data
            self.grads_data = ret[1:]

        # return err_list
        return err_list

    def doTrain(self, _batch_tuples, _weights):
        # get inputs from batch
        cur_x = self.getCurInputs(_batch_tuples)
        next_x = self.getNextInputs(_batch_tuples)
        # compute forward
        next_output = self.forward(next_x)
        # fill grad
        err_list = self.grad(cur_x, next_output, _batch_tuples, _weights)

        return err_list

    def chooseAction(self, _model, _state):
        x_data = self.env.getX(_state)
        output = self.func(_model, x_data, False)
        logger.info(output)
        if self.is_train:
            return self.env.getSoftAction(output, [_state])[0]
        else:
            return self.env.getBestAction(output, [_state])[0]
