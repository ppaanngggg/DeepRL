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
        # whether to clip err
        self.err_clip = None
        # whether to clip grad
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

        self.epoch_show_log = None


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
        self.target_v_func = None

        self.q_func = None
        self.target_q_func = None

        self.p_func = None
        self.target_p_func = None

        # alloc train part
        self.vars = None
        self.set_vars_op = None
        self.set_vars_place = None

        self.target_vars = None
        self.set_target_vars_op = None
        self.set_target_vars_place = None

        self.grads_data = None
        self.grads_place = None

        # alloc opt
        self.opt = None

        # alloc env
        self.env = None

        # is train
        self.is_train = _is_train

        # set config, and set device
        self.config = Config()
        self.config.device = '/gpu:0' if _gpu else '/cpu:0'

        # set place for x
        with tf.device(self.config.device):
            self.x_place = tf.placeholder(tf.float32)

        # count new game times
        self.epoch = 0

    def createOpt(self, _opt, _global_step):
        self.grads_place = [
            tf.placeholder(tf.float32) for _ in self.vars
        ]
        if self.config.grad_clip:
            grads_op = [
                tf.clip_by_value(
                    d, -self.config.grad_clip, self.config.grad_clip)
                for d in self.grads_place
            ]
        else:
            grads_op = self.grads_place
        self.opt = _opt.apply_gradients([
            (p, v) for p, v in zip(grads_op, self.vars)
        ], global_step=_global_step)

    def getVars(self):
        if self.vars:
            return self.sess.run(self.vars)
        return None

    def getTargetVars(self):
        if self.target_vars:
            return self.sess.run(self.target_vars)
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

    def setVars(self, _data):
        if self.vars:
            if self.set_vars_op is None and self.set_vars_place is None:
                self.set_vars_op, self.set_vars_place = \
                    self.createSetOpPlace(self.vars)
            self.sess.run(
                self.set_vars_op,
                feed_dict=self.createSetFeedDict(
                    self.set_vars_place, _data)
            )

    def setTargetVars(self, _data):
        if self.target_vars:
            if self.set_target_vars_op is None and self.set_target_vars_place is None:
                self.set_target_vars_op, self.set_target_vars_place = \
                    self.createSetOpPlace(self.target_vars)
            self.sess.run(
                self.set_target_vars_op,
                feed_dict=self.createSetFeedDict(
                    self.set_target_vars_place, _data)
            )

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
            self.env.startNewGame()
        self.epoch += 1
        if self.epoch % self.config.epoch_show_log == 0:
            logger.info('start new game')

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

        if self.epoch % self.config.epoch_show_log == 0:
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
                if self.epoch % self.config.epoch_show_log == 0:
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

    def stepUntilEnd(self, _model):
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
            while self.env.in_game:
                state_list.append(cur_state)
                action = self.chooseAction(_model, cur_state)
                action_list.append(action)
                reward = self.env.doAction(action)
                reward_list.append(reward)
                if self.epoch % self.config.epoch_show_log == 0:
                    logger.info(
                        'Action: ' + str(action) + '; Reward: %.3f' % (reward))
                next_state = self.env.getState()
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
        def apply_grads(_opt, _places, _vars, _grads):
            if _opt is not None and _places and _vars and _grads:
                tmp = {}
                for p, g in zip(_places, _grads):
                    tmp[p] = g
                self.sess.run(_opt, feed_dict=tmp)
        with tf.device(self.config.device):
            apply_grads(self.opt, self.grads_place, self.vars, self.grads_data)

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
        with tf.device(self.config.device):
            if self.vars and self.target_vars:
                self.setTargetVars(self.getVars())

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
                if self.epoch % self.config.epoch_show_log == 0:
                    logger.info(str(output))
                return self.env.getBestAction(output, [_state])[0]
        else:
            x_data = self.env.getX(_state)
            output = self.func(_model, x_data, False)
            logger.info(str(output))
            return self.env.getBestAction(output, [_state])[0]

    def chooseSoftAction(self, _model, _state):
        x_data = self.env.getX(_state)
        output = self.func(_model, x_data, False)

        if self.is_train:
            if self.epoch % self.config.epoch_show_log == 0:
                logger.info(output)
            return self.env.getSoftAction(output, [_state])[0]
        else:
            logger.info(output)
            return self.env.getBestAction(output, [_state])[0]

    def save(self, _epoch, _step):
        filename = './models/epoch_' + str(_epoch) + '_step_' + str(_step)
        logger.info(filename)
        with tf.device(self.config.device):
            np.save(filename, self.getVars())

    def load(self, filename):
        logger.info(filename)
        with tf.device(self.config.device):
            self.setVars(np.load(filename))

        self.updateTargetFunc()
