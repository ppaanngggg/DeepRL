from Agent import Agent
import random
import tensorflow as tf
import numpy as np


class QAgent(Agent):
    """
    Human-level control through deep reinforcement learning

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
        _epsilon (float): init epsilon, p for choosing randomly
        _epsilon_decay (float): epsilon *= epsilon_decay
        _epsilon_underline (float): epsilon = max(epsilon_underline, epsilon)
        _grad_clip (float): clip grad, 0 is no clip
    """

    def __init__(self, _model, _env, _is_train=True,
                 _optimizer=None, _global_step=None, _replay=None,
                 _gpu=False, _gamma=0.99, _batch_size=32,
                 _epsilon=0.5, _epsilon_decay=0.995, _epsilon_underline=0.01,
                 _err_clip=None, _grad_clip=None, _epoch_show_log=1e3):

        super(QAgent, self).__init__(_is_train, _gpu)

        # set config
        self.config.gamma = _gamma
        self.config.batch_size = _batch_size
        self.config.epsilon = _epsilon
        self.config.epsilon_decay = _epsilon_decay
        self.config.epsilon_underline = _epsilon_underline
        self.config.err_clip = _err_clip
        self.config.grad_clip = _grad_clip
        self.config.epoch_show_log = _epoch_show_log

        # set env
        self.env = _env

        with tf.device(self.config.device):
            # create q func
            self.q_func, self.vars = _model(self.x_place)

            if self.is_train:
                # create target q func
                self.target_q_func, self.target_vars = _model(self.x_place)
                # place for action(one hot), target, weight
                self.action_place = tf.placeholder(tf.float32)
                self.target_place = tf.placeholder(tf.float32)
                self.weight_place = tf.placeholder(tf.float32)
                # get cur action value
                action_value = tf.reduce_sum(
                    self.q_func * self.action_place, 1
                )
                # get err of cur action value and target value
                self.err_list_op = 0.5 * \
                    tf.square(action_value - self.target_place)
                # clipped err
                if self.config.err_clip:
                    self.clipped_err_op = tf.clip_by_value(
                        self.err_list_op,
                        -self.config.err_clip, self.config.err_clip
                    )
                else:
                    self.clipped_err_op = self.err_list_op
                # get total loss, mul with weight, if weight exist
                loss = tf.reduce_mean(self.clipped_err_op * self.weight_place)
                # compute grads of vars
                self.grads_op = tf.gradients(loss, self.vars)

                if _optimizer:
                    self.createOpt(_optimizer, _global_step)

                self.replay = _replay

            # init all vars
            self.sess.run(tf.initialize_all_variables())
            # copy params from q func to target
            self.updateTargetFunc()

    def step(self):
        """
        Returns:
            still in game or not
        """
        return super(QAgent, self).step(self.q_func)

    def forward(self, _next_x, _state_list):
        with tf.device(self.config.device):
            # get next outputs, NOT target
            next_output = self.func(self.q_func, _next_x, False)
            # only one head
            next_action = self.env.getBestAction(next_output, _state_list)
            # get next outputs, target
            next_output = self.func(self.target_q_func, _next_x, False)

        return next_output, next_action

    def grad(self, _cur_x, _next_output, _next_action, _batch_tuples, _weights):
        with tf.device(self.config.device):
            # get action data (one hot)
            action_data = self.getActionData(
                self.q_func.get_shape().as_list()[1], _batch_tuples)
            # get target data
            target_data = self.getQTargetData(
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
            self.grads_data = ret[2:]
        # return err_list
        return ret[0]

    def doTrain(self, _batch_tuples, _weights):
        # get inputs from batch
        cur_x = self.getCurInputs(_batch_tuples)
        next_x = self.getNextInputs(_batch_tuples)
        # compute forward
        next_output, next_action = self.forward(
            next_x, [t.next_state for t in _batch_tuples])
        # fill grad
        err_list = self.grad(
            cur_x, next_output, next_action, _batch_tuples, _weights)
        return err_list
