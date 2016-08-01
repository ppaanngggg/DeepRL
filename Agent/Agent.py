import random
import tensorflow as tf
import numpy as np

import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


class Config(object):
    """
    Config for Agent
    """

    def __init__(self):
        """
        Alloc all configs
        """
        # whether to use gpu
        self.device = None
        # gamma, decay param of reward
        self.gamma = None
        # batch size of train
        self.batch_size = None
        # randomly choose action
        self.epsilon = None
        self.epsilon_decay = None
        self.epsilon_underline = None
        # where to clip grad
        self.grad_clip = None
        # bootstrapped, num of heads
        self.K = None
        # bootstrapped, p of mask
        self.mask_p = None
        # NFSP when to use p_func
        self.eta = None
        # for N step
        self.step_len = None
        # beta of entropy
        self.beta_entropy = None


class Agent(object):
    """
    base class of other agents
    """

    def __init__(self, _is_train, _gpu):
        tmp = tf.ConfigProto()
        tmp.gpu_options.allow_growth = True
        self.sess = tf.Session(config=tmp)

        # alloc all models
        self.v_func = None
        self.v_vars = None
        self.set_v_vars_op = None
        self.set_v_vars_place = None
        self.v_grads_data = None
        self.v_grads_place = None
        self.target_v_func = None
        self.target_v_vars = None
        self.set_target_v_vars_op = None
        self.set_target_v_vars_place = None

        self.q_func = None
        self.q_vars = None
        self.set_q_vars_op = None
        self.set_q_vars_place = None
        self.q_grads_data = None
        self.q_grads_place = None
        self.target_q_func = None
        self.target_q_vars = None
        self.set_target_q_vars_op = None
        self.set_target_q_vars_place = None

        self.p_func = None
        self.p_vars = None
        self.set_p_vars_op = None
        self.set_p_vars_place = None
        self.p_grads_data = None
        self.p_grads_place = None
        self.target_p_func = None
        self.target_p_vars = None
        self.set_target_p_vars_op = None
        self.set_target_p_vars_place = None

        # alloc all optimizers
        self.v_opt = None
        self.q_opt = None
        self.p_opt = None

        self.env = None

        # is train
        self.is_train = _is_train

        # set config, and set device
        self.config = Config()
        self.config.device = '/gpu:0' if _gpu else '/cpu:0'

        # set place for x
        with tf.device(self.config.device):
            self.x_place = tf.placeholder(tf.float32)

    def getVFunc(self):
        if self.v_vars:
            return self.sess.run(self.v_vars)
        return None

    def getTargetVFunc(self):
        if self.target_v_vars:
            return self.sess.run(self.target_v_vars)
        return None

    def getQFunc(self):
        if self.q_vars:
            return self.sess.run(self.q_vars)
        return None

    def getTargetQFunc(self):
        if self.target_q_vars:
            return self.sess.run(self.target_q_vars)
        return None

    def getPFunc(self):
        if self.p_vars:
            return self.sess.run(self.p_vars)
        return None

    def getTargetPFunc(self):
        if self.target_p_vars:
            return self.sess.run(self.target_p_vars)
        return None

    def createSetOpPlace(self, _vars):
        place = [tf.placeholder(tf.float32) for _ in _vars]
        op = [v.assign(p) for v, p in zip(_vars, place)]
        return op, place

    def createSetFeedDict(self, _place, _data):
        tmp = {}
        for p, d in zip(_place, _data):
            tmp[p] = d
        return tmp

    def setVFunc(self, _data):
        if self.v_vars:
            if self.set_v_vars_op is None and self.set_v_vars_place is None:
                self.set_v_vars_op, self.set_v_vars_place = \
                    self.createSetOpPlace(self.v_vars)
            self.sess.run(
                self.set_v_vars_op,
                feed_dict=self.createSetFeedDict(self.set_v_vars_place, _data))

    def setTargetVFunc(self, _data):
        if self.target_v_vars:
            tmp = {}
            for p, d in zip(self.set_target_v_vars_place, _data):
                tmp[p] = d
            self.sess.run(self.set_target_v_vars_op, feed_dict=tmp)

    def setQFunc(self, _data):
        if self.q_vars:
            tmp = {}
            for p, d in zip(self.set_q_vars_place, _data):
                tmp[p] = d
            self.sess.run(self.set_target_q_vars_op, feed_dict=tmp)

    def setTargetQFunc(self, _data):
        if self.target_q_vars:

    def training(self):
        """
        set agent to train mod
        """
        self.is_train = True

    def evaluating(self):
        """
        set agent to evaluate mod
        """
        self.is_train = False

    def startNewGame(self):
        """
        normal start new game, suitable for most agent
        """
        while not self.env.in_game:
            logger.info('Env not in game')
            self.env.startNewGame()

    def step(self, _model):
        """
        agent will get cur state and choose one action and execute

        Returns:
            still in game or not
        """
        if not self.env.in_game:
            return False

        # get current state
        cur_state = self.env.getState()
        # choose action in step
        action = self.chooseAction(_model, cur_state)
        # do action and get reward
        reward = self.env.doAction(action)

        logger.info('Action: ' + str(action) + '; Reward: %.3f' % (reward))

        if self.is_train:
            # get new state
            next_state = self.env.getState()
            # store replay_tuple into memory pool
            self.replay.push(cur_state, action, reward, next_state)

        return self.env.in_game

    def nstep(self, _model):
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
                action = self.chooseAction(_model, cur_state)
                action_list.append(action)
                reward = self.env.doAction(action)
                reward_list.append(reward)
                logger.info(
                    'Action: ' + str(action) + '; Reward: %.3f' % (reward))
                next_state = self.env.getState()
                if not self.env.in_game:
                    break
                cur_state = next_state

            for i in range(len(state_list)):
                self.replay.push(
                    state_list[i], action_list[i], reward_list[i:], next_state)
            return self.env.in_game
        else:
            return self.step(_model)

    def train(self):
        """
        train model
        """
        # pull tuples from memory pool
        batch_tuples, weights = self.replay.pull(self.config.batch_size)
        if not len(batch_tuples):
            return

        err_list = self.doTrain(batch_tuples, weights)

        self.update()

        # set err and merge
        self.replay.setErr(batch_tuples, err_list)
        self.replay.merge()

    def update(self):
        # if has optimizers, then update model
        def apply_grads(_opt, _func, _places, _vars, _grads):
            if _opt is not None and _func is not None \
                    and _places and _vars and _grads:
                tmp = {}
                for p, g in zip(_places, _grads):
                    tmp[p] = g
                self.sess.run(_opt, feed_dict=tmp)
        with tf.device(self.config.device):
            apply_grads(self.v_opt, self.v_func, self.v_grads_place,
                        self.v_vars, self.v_grads_data)
            apply_grads(self.q_opt, self.q_func, self.q_grads_place,
                        self.q_vars, self.q_grads_data)
            apply_grads(self.p_opt, self.p_func, self.p_grads_place,
                        self.p_vars, self.p_grads_data)

    def doTrain(self, _batch_tuples, _weights):
        """
        do train detail, need to be overwritten
        """
        raise Exception()

    def getCurInputs(self, _batch_tuples):
        """
        get and stack cur inputs from tuples
        """
        # stack inputs
        cur_x = [self.env.getX(t.state) for t in _batch_tuples]
        # merge inputs into one array
        cur_x = np.concatenate(cur_x, 0)
        return cur_x

    def getNextInputs(self, _batch_tuples):
        """
        get and stack next inputs from tuples
        """
        # stack inputs
        next_x = [self.env.getX(t.next_state) for t in _batch_tuples]
        # merge inputs into one array
        next_x = np.concatenate(next_x, 0)
        return next_x

    def getActionData(self, _action_space, _batch_tuples):
        action_data = np.zeros((len(_batch_tuples), _action_space))
        for i in range(len(_batch_tuples)):
            action_data[i, _batch_tuples[i].action] = 1.
        return action_data

    def getQTargetData(self, _next_output, _next_action, _batch_tuples):
        target_data = np.zeros((len(_batch_tuples)), np.float32)
        for i in range(len(_batch_tuples)):
            target_value = _batch_tuples[i].reward
            # if not empty position, not terminal state
            if _batch_tuples[i].next_state.in_game:
                target_value += self.config.gamma * \
                    _next_output[i][_next_action[i]]
            target_data[i] = target_value
        return target_data

    def getNStepQTargetData(self, _next_output, _next_action, _batch_tuples):
        target_data = np.zeros((len(_batch_tuples)), np.float32)
        for i in range(len(_batch_tuples)):
            reward = _batch_tuples[i].reward
            target_value = 0.
            # if not empty position, not terminal state
            if _batch_tuples[i].next_state.in_game:
                target_value += _next_output[i][_next_action[i]]
            for r in reversed(reward):
                target_value = r + self.config.gamma * target_value
            target_data[i] = target_value
        return target_data

    def getVTargetData(self, _next_output, _batch_tuples):
        target_data = np.zeros((len(_batch_tuples)), np.float32)
        for i in range(len(_batch_tuples)):
            target_value = _batch_tuples[i].reward
            # if not empty position, not terminal state
            if _batch_tuples[i].next_state.in_game:
                target_value += self.config.gamma * _next_output[i][0]
            target_data[i] = target_value
        return target_data

    def getNStepVTargetData(self, _next_output, _batch_tuples):
        target_data = np.zeros((len(_batch_tuples)), np.float32)
        for i in range(len(_batch_tuples)):
            reward = _batch_tuples[i].reward
            target_value = 0.
            # if not empty position, not terminal state
            if _batch_tuples[i].next_state.in_game:
                target_value += _next_output[i][0]
            for r in reversed(reward):
                target_value = r + self.config.gamma * target_value
            target_data[i] = target_value
        return target_data

    def getWeightData(self, _weights, _batch_tuples):
        if _weights is not None:
            return _weights
        else:
            return np.ones((len(_batch_tuples)), np.float32)

    def func(self, _model, _x_data, _train=True):
        """
        do func use special model with input
        """
        with tf.device(self.config.device):
            ret = self.sess.run(_model, feed_dict={self.x_place: _x_data})
        return ret

    def updateTargetFunc(self):
        """
        update target if exit
        """
        logger.info('update target func')

        def assign_vars(_s, _d):
            if _s and _d:
                for _s_v, _d_v in zip(_s, _d):
                    self.sess.run(_d_v.assign(_s_v))

        with tf.device(self.config.device):
            assign_vars(self.v_vars, self.target_v_vars)
            assign_vars(self.q_vars, self.target_q_vars)
            assign_vars(self.p_vars, self.target_p_vars)

    def updateEpsilon(self):
        """
        update epsilon
        """
        self.config.epsilon = max(
            self.config.epsilon_underline,
            self.config.epsilon * self.config.epsilon_decay
        )

    def chooseAction(self, _model, _state):
        """
        choose action by special model in special state, suitable for most agent
        """
        if self.is_train:
            # update epsilon
            self.updateEpsilon()
            random_value = random.random()
            if random_value < self.config.epsilon:
                # randomly choose
                return self.env.getRandomAction(_state)
            else:
                # use model to choose
                x_data = self.env.getX(_state)
                output = self.func(_model, x_data, False)
                logger.info(str(output))
                return self.env.getBestAction(output, [_state])[0]
        else:
            x_data = self.env.getX(_state)
            output = self.func(_model, x_data, False)
            logger.info(str(output))
            return self.env.getBestAction(output, [_state])[0]

    def save(self, _epoch, _step):
        filename = './models/epoch_' + str(_epoch) + '_step_' + str(_step)
        logger.info(filename)
        if self.v_vars:
            np.save(
                filename + '_v_func',
                [self.sess.run(v) for v in self.v_vars]
            )
        if self.q_vars:
            np.save(
                filename + '_q_func',
                [self.sess.run(v) for v in self.q_vars]
            )
        if self.p_vars:
            np.save(
                filename + '_p_func',
                [self.sess.run(v) for v in self.p_vars]
            )

    def load(self, filename):
        logger.info(filename)
        if self.v_vars:
            with tf.device(self.config.device):
                for d, v in zip(np.load(filename + '_v_func.npy'), self.v_vars):
                    self.sess.run(v.assign(d))
        if self.q_vars:
            with tf.device(self.config.device):
                for d, v in zip(np.load(filename + '_q_func.npy'), self.q_vars):
                    self.sess.run(v.assign(d))
        if self.p_vars:
            with tf.device(self.config.device):
                for d, v in zip(np.load(filename + '_p_func.npy'), self.p_vars):
                    self.sess.run(v.assign(d))

        self.updateTargetFunc()
