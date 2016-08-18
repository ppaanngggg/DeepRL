from Agent import Agent
import random
import tensorflow as tf
import numpy as np

import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


class QACAgent(Agent):
    """
    Determinisitc Policy Gradient Algorithms

    Args:
        _actor (function): necessary, head part of p func,
                        output's dim should be equal with num of actions
        _critic (function): necessary, head part of q func,
                        output's dim should be equal with num of actions
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
        _beta_entropy (float): beta for entropy
        _grad_clip (float): clip grad, 0 is no clip
    """

    def __init__(self, _actor, _critic, _env, _is_train=True,
                 _actor_optimizer=None, _critic_optimizer=None, _replay=None,
                 _gpu=False, _gamma=0.99, _batch_size=32, _beta_entropy=0.01,
                 _grad_clip=1.):

        super(QACAgent, self).__init__(_is_train, _gpu)

        # set env
        self.env = _env

        with tf.device(self.config.device):
            # create p func
            self.p_func, self.p_vars = _actor(self.x_place)
            # get softmax of p func
            self.softmax_op = tf.nn.softmax(self.p_func)
            self.q_func, self.q_vars = _critic(self.x_place)

            if self.is_train:
                # critic part :
                # place for action(one hot), target, weight
                self.action_place = tf.placeholder(tf.float32)
                self.target_place = tf.placeholder(tf.float32)
                self.weight_place = tf.placeholder(tf.float32)
                # get cur action value
                self.action_value_op = tf.reduce_sum(
                    self.q_func * self.action_place, 1
                )
                # get err of cur action value and target value
                self.err_list_op = 0.5 * \
                    tf.square(self.action_value_op - self.target_place)
                # get total loss, mul with weight, if weight exist
                loss = tf.reduce_mean(self.err_list_op * self.weight_place)
                # compute grads of vars
                self.critic_grads_op = tf.gradients(loss, self.q_vars)

                # actor part :
                # place for action, critic_place
                self.critic_place = tf.placeholder(tf.float32)
                # get entropy
                entropy = - tf.reduce_sum(
                    self.softmax_op * tf.log(self.softmax_op + 1e-10))
                # get loss
                loss = tf.reduce_sum(
                    -tf.log(
                        tf.reduce_sum(self.softmax_op * self.action_place, 1)
                        + 1e-10
                    ) * self.critic_place
                ) + _beta_entropy * entropy
                # compute grads of vars
                self.actor_grads_op = tf.gradients(loss, self.p_vars)

                if _actor_optimizer:
                    self.createPOpt(_actor_optimizer)
                if _critic_optimizer:
                    self.createQOpt(_critic_optimizer)

                self.replay = _replay

            # init all vars
            self.sess.run(tf.initialize_all_variables())

        self.config.gpu = _gpu
        self.config.gamma = _gamma
        self.config.batch_size = _batch_size
        self.config.grad_clip = _grad_clip

    def step(self):
        """
        Returns:
            still in game or not
        """
        return super(QACAgent, self).step(self.p_func)

    def forward(self, _next_x, _state_list):
        with tf.device(self.config.device):
            # get next outputs
            next_output = self.func(self.q_func, _next_x, False)

            tmp_next_output = self.func(self.softmax_op, _next_x, False)
            next_action = self.env.getBestAction(tmp_next_output, _state_list)

        return next_output, next_action

    def grad(self, _cur_x, _next_output, _next_action, _batch_tuples):
        with tf.device(self.config.device):
            # critic part
            # get action data
            action_data = self.getActionData(
                self.q_func.get_shape().as_list()[1], _batch_tuples)
            # get target data
            target_data = self.getQTargetData(
                _next_output, _next_action, _batch_tuples)
            # get weight data
            weight_data = self.getWeightData(_weights, _batch_tuples)
            # get actor_value [0], err_list [1] and grads [2:]
            ret = self.sess.run(
                [self.action_value_op, self.err_list_op] + self.critic_grads_op,
                feed_dict={
                    self.x_place: _cur_x,
                    self.action_place: action_data,
                    self.target_place: target_data,
                    self.weight_place: weight_data,
                }
            )
            critic_data = ret[0]
            err_list = ret[1]
            # set q grad data
            self.q_grads_data = ret[2:]

            # actor part
            # set p grads data
            self.p_grads_data = self.sess.run(
                self.actor_grads_op,
                feed_dict={
                    self.x_place: _cur_x,
                    self.action_place: action_data,
                    self.critic_place: critic_data
                }
            )

        return err_list

    def doTrain(self, _batch_tuples, _weights):
        # get inputs from batch
        cur_x = self.getCurInputs(_batch_tuples)
        next_x = self.getNextInputs(_batch_tuples)
        # compute forward
        next_output, next_action = self.forward(
            next_x, [t.next_state for t in _batch_tuples])
        # fill grad
        err_list = self.grad(
            cur_x, next_output, next_action, _batch_tuples)

        return err_list

    def chooseAction(self, _model, _state):
        x_data = self.env.getX(_state)
        output = self.func(_model, x_data, False)
        logger.info(str(F.softmax(output).data))
        if self.is_train:
            return self.env.getSoftAction(output.data, [_state])[0]
        else:
            return self.env.getBestAction(output.data, [_state])[0]
