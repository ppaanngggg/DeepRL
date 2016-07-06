import random
from chainer import serializers, optimizers, Variable
from chainer import cuda
import numpy as np

import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


class Agent(object):

    def training(self):
        self.is_train = True

    def testing(self):
        self.is_train = False

    def startNewGame(self):
        while not self.env.in_game:
            logger.info('Env not in game')
            self.env.startNewGame()

    def getInputs(self, _batch_tuples):
        # stack inputs
        cur_x = [self.env.getX(t.state) for t in _batch_tuples]
        next_x = [self.env.getX(t.next_state) for t in _batch_tuples]
        # merge inputs into one array
        if self.gpu:
            cur_x = cupy.concatenate(cur_x, 0)
            next_x = cupy.concatenate(next_x, 0)
        else:
            cur_x = np.concatenate(cur_x, 0)
            next_x = np.concatenate(next_x, 0)
        return cur_x, next_x

    def gradWeight(self, _cur_output, _weights):
        # multiply weights with grad
        if self.gpu:
            _cur_output.grad = cupy.multiply(
                _cur_output.grad, _weights)
        else:
            _cur_output.grad = np.multiply(
                _cur_output.grad, _weights)

    def gradClip(self, _cur_output, _value=1):
        # clip grads
        if self.gpu:
            _cur_output.grad = cupy.clip(
                _cur_output.grad, -_value, _value)
        else:
            _cur_output.grad = np.clip(
                _cur_output.grad, -_value, _value)

    def toVariable(self, _data):
        if type(_data) is list:
            return [self.toVariable(d) for d in _data]
        else:
            return Variable(_data)

    def func(self, _model, _x_data, _train=True):
        if _train:
            _model.training()
        else:
            _model.evaluating()
        return _model(self.toVariable(_x_data))

    def updateTargetQFunc(self):
        logger.info('update target func')
        self.target_q_func.copyparams(self.q_func)

    def save(self, _epoch, _step):
        filename = './models/epoch_' + str(_epoch) + '_step_' + str(_step)
        logger.info(filename)
        serializers.save_npz(filename, self.q_func)

    def load(self, filename):
        logger.info(filename)
        serializers.load_npz(filename, self.q_func)
        if self.is_train:
            self.target_q_func.copyparams(self.q_func)
