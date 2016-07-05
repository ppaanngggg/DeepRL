import logging
import numpy as np
from chainer import cuda


class State(object):

    def __init__(self, _in_game, _dict):
        self.in_game = _in_game
        if type(_dict) is dict:
            self.state = _dict
        else:
            raise Exception()

    def show(self):
        print '##### State #####'
        print '## in_game :', self.in_game
        for k, v in zip(self.state.keys(), self.state.values()):
            print '##', k, ':', v
        print '#################'


class Env(object):

    def __init__(self):
        self.in_game = False

    def startNewGame(self):
        self.in_game = True
        logging.info('Start new game')
        self.doStartNewGame()

    def getState(self):
        return State(self.in_game, self.doGetState())

    def doAction(self, _action):
        return self.doDoAction(_action)

    def getX(self, _state):
        ret = self.doGetX(_state)
        if type(ret) is not np.ndarray and type(ret) is not cuda.ndarray:
            raise Exception()
        return ret

    def getRandomAction(self, _state):
        ret = self.doGetRandomAction(_state)
        if type(ret) is not int:
            raise Exception()
        return ret

    def getBestAction(self, _data, _state_list):
        ret = self.doGetBestAction(_data, _state_list)
        if type(ret) is not list:
            raise Exception()
        for d in ret:
            if type(d) is not int:
                raise Exception()
        return ret

    # need to be overwritten by user
    def doStartNewGame(self):
        raise Exception()

    def doGetState(self):
        raise Exception()

    def doDoAction(self, _action):
        raise Exception()

    def doGetX(self, _state):
        raise Exception()

    def doGetRandomAction(self, _state):
        raise Exception()

    def doGetBestAction(self, _data, _state_list):
        raise Exception()
