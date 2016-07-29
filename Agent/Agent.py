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
        self.gpu = None
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


class Agent(object):
    """
    base class of other agents
    """

    def __init__(self):
        # alloc all models
        self.v_func = None
        self.v_vars = None
        self.v_grads_data = None
        self.v_grads_place = None
        self.target_v_func = None
        self.target_v_vars = None

        self.q_func = None
        self.q_vars = None
        self.q_grads_data = None
        self.q_grads_place = None
        self.target_q_func = None
        self.target_q_vars = None

        self.p_func = None
        self.p_vars = None
        self.p_grads_data = None
        self.p_grads_place = None
        self.target_p_func = None
        self.target_p_vars = None
        # alloc all optimizers
        self.v_opt = None
        self.q_opt = None
        self.p_opt = None

        self.env = None

        self.config = Config()

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

    def train(self):
        """
        train model
        """
        # pull tuples from memory pool
        batch_tuples, weights = self.replay.pull(self.config.batch_size)
        if not len(batch_tuples):
            return

        err_list = self.doTrain(batch_tuples, weights)

        # if has optimizers, then update model
        def apply_grads(_opt, _func, _places, _vars, _grads):
            if _opt is not None and _func is not None \
                    and _places and _vars and _grads:
                tmp = {}
                for p, g in zip(_places, _grads):
                    tmp[p] = g
                self.sess.run(_opt, feed_dict=tmp)
        with tf.device(self.device):
            apply_grads(self.v_opt, self.v_func, self.v_grads_place,
                        self.v_vars, self.v_grads_data)
            apply_grads(self.q_opt, self.q_func, self.q_grads_place,
                        self.q_vars, self.q_grads_data)
            apply_grads(self.p_opt, self.p_func, self.p_grads_place,
                        self.p_vars, self.p_grads_data)

        # set err and merge
        self.replay.setErr(batch_tuples, err_list)
        self.replay.merge()

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

    def gradWeight(self, _variable, _weights):
        """
        multiply grad with weights
        """
        # multiply weights with grad
        if self.config.gpu:
            _variable.grad = cupy.multiply(_variable.grad, _weights)
        else:
            _variable.grad = np.multiply(_variable.grad, _weights)

    def gradClip(self, _variable, _value=1):
        """
        clip grad with limit
        """
        # clip grads
        if self.config.gpu:
            _variable.grad = cupy.clip(_variable.grad, -_value, _value)
        else:
            _variable.grad = np.clip(_variable.grad, -_value, _value)

    def func(self, _model, _x_data, _train=True):
        """
        do func use special model with input
        """
        with tf.device(self.device):
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

        with tf.device(self.device):
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
        if self.v_func:
            serializers.load_npz(filename + '_v_func', self.v_func)
        if self.q_func:
            serializers.load_npz(filename + '_q_func', self.q_func)
        if self.p_func:
            serializers.load_npz(filename + '_p_func', self.p_func)

        self.updateTargetFunc()
