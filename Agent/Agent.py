import random
from chainer import serializers, optimizers, Variable
from chainer import cuda
try:
    import cupy
except:
    pass
import numpy as np

import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


class Config(object):

    def __init__(self):
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


class Agent(object):

    def __init__(self):
        self.v_func = None
        self.target_v_func = None
        self.q_func = None
        self.target_q_func = None
        self.p_func = None
        self.target_p_func = None

        self.v_opt = None
        self.q_opt = None
        self.p_opt = None

        self.env = None

        self.config = Config()

    def training(self):
        self.is_train = True

    def evaluating(self):
        self.is_train = False

    def startNewGame(self):
        while not self.env.in_game:
            logger.info('Env not in game')
            self.env.startNewGame()

    def step(self, _model):
        """
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
        # clear grads
        if self.v_func:
            self.v_func.zerograds()
        if self.q_func:
            self.q_func.zerograds()
        if self.p_func:
            self.p_func.zerograds()
        # pull tuples from memory pool
        batch_tuples, weights = self.replay.pull(self.config.batch_size)
        if not len(batch_tuples):
            return

        err_list = self.doTrain(batch_tuples, weights)

        if self.v_opt and self.v_func:
            self.v_opt.update()
        if self.q_opt and self.q_func:
            self.q_opt.update()
        if self.p_opt and self.p_func:
            self.p_opt.update()

        self.replay.setErr(batch_tuples, err_list)
        self.replay.merge()

    def doTrain(self, _batch_tuples, _err_list):
        raise Exception()

    def getCurInputs(self, _batch_tuples):
        # stack inputs
        cur_x = [self.env.getX(t.state) for t in _batch_tuples]
        # merge inputs into one array
        cur_x = np.concatenate(cur_x, 0)
        return cur_x

    def getNextInputs(self, _batch_tuples):
        # stack inputs
        next_x = [self.env.getX(t.next_state) for t in _batch_tuples]
        # merge inputs into one array
        next_x = np.concatenate(next_x, 0)
        return next_x

    def gradWeight(self, _variable, _weights):
        # multiply weights with grad
        if self.config.gpu:
            _variable.grad = cupy.multiply(_variable.grad, _weights)
        else:
            _variable.grad = np.multiply(_variable.grad, _weights)

    def gradClip(self, _variable, _value=1):
        # clip grads
        if self.config.gpu:
            _variable.grad = cupy.clip(_variable.grad, -_value, _value)
        else:
            _variable.grad = np.clip(_variable.grad, -_value, _value)

    def toGPU(self, _data):
        if type(_data) is list:
            return [self.toGPU(d) for d in _data]
        else:
            return cuda.to_gpu(_data)

    def toVariable(self, _data):
        if type(_data) is list:
            return [self.toVariable(d) for d in _data]
        else:
            return Variable(_data)

    def func(self, _model, _x_data, _train=True):
        if self.config.gpu:
            _x_data = self.toGPU(_x_data)
        if _train:
            _model.training()
        else:
            _model.evaluating()
        return _model(self.toVariable(_x_data))

    def updateTargetFunc(self):
        logger.info('update target func')
        if self.target_v_func and self.v_func:
            self.target_v_func.copyparams(self.v_func)
        if self.target_q_func and self.q_func:
            self.target_q_func.copyparams(self.q_func)
        if self.target_p_func and self.p_func:
            self.target_p_func.copyparams(self.p_func)

    def updateEpsilon(self):
        self.config.epsilon = max(
            self.config.epsilon_underline,
            self.config.epsilon * self.config.epsilon_decay
        )

    def chooseAction(self, _model, _state):
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
                print output.data
                return self.env.getBestAction(output.data, [_state])[0]
        else:
            x_data = self.env.getX(_state)
            output = self.func(_model, x_data, False)
            logger.info(str(output.data))
            return self.env.getBestAction(output.data, [_state])[0]

    def save(self, _epoch, _step):
        filename = './models/epoch_' + str(_epoch) + '_step_' + str(_step)
        logger.info(filename)
        if self.v_func:
            serializers.save_npz(filename + '_v_func', self.v_func)
        if self.q_func:
            serializers.save_npz(filename + '_q_func', self.q_func)
        if self.p_func:
            serializers.save_npz(filename + '_p_func', self.p_func)

    def load(self, filename):
        logger.info(filename)
        if self.v_func:
            serializers.load_npz(filename + '_v_func', self.v_func)
        if self.q_func:
            serializers.load_npz(filename + '_q_func', self.q_func)
        if self.p_func:
            serializers.load_npz(filename + '_p_func', self.p_func)

        if self.target_v_func:
            self.target_v_func.copyparams(self.v_func)
        if self.target_q_func:
            self.target_q_func.copyparams(self.q_func)
        if self.target_p_func:
            self.target_p_func.copyparams(self.p_func)

    def copyFunc(_agent):
        if self.v_func and _agent.v_func:
            self.v_func.copyparams(_agent.v_func)
        if self.q_func and _agent.q_func:
            self.q_func.copyparams(_agent.q_func)
        if self.p_func and _agent.p_func:
            self.p_func.copyparams(_agent.p_func)
        if self.target_v_func and _agent.target_v_func:
            self.target_v_func.copyparams(_agent.target_v_func)
        if self.target_q_func and _agent.target_q_func:
            self.target_q_func.copyparams(_agent.target_q_func)
        if self.target_p_func and _agent.target_p_func:
            self.target_p_func.copyparams(_agent.target_p_func)
