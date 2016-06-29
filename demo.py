from Agent import *
from Env import Env
import Config
from Replay import *
from Train import Train
import logging
import random
import numpy as np
import chainer.links as L

FORMAT = '%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - %(message)s'
logging.basicConfig(format=FORMAT, level=logging.DEBUG)


class DemoEnv(Env):

    def __init__(self):
        super(DemoEnv, self).__init__()

    def doStartNewGame(self):
        self.loc = 1
        self.count = 0

    def doGetState(self):
        return {'loc': self.loc}

    def doDoAction(self, _action):
        if _action == 1:  # a difficlut case
            if random.random() < 0.5:
                _action = 0
        if _action == 0:  # go left
            self.loc = max(0, self.loc - 1)
        if _action == 1:  # go right
            self.loc = min(5, self.loc + 1)

        self.count += 1
        if self.count >= 15:
            self.in_game = False

        if self.loc == 0:
            return 0.001
        if self.loc == 5:
            return 1
        return 0

    def doGetX(self, _state):
        return np.array([[_state.state['loc']]], np.float32)

    def doGetRandomAction(self, _state):
        return random.randint(0, 1)

    def doGetBestAction(self, _data, _state_list):
        return np.argmax(_data, 1).tolist()


class Shared(Chain):

    def __init__(self):
        super(Shared, self).__init__(
            linear=L.Linear(1, 5)
        )

    def __call__(self, _x, _is_train):
        y = self.linear(_x)
        return y


class Head(Chain):

    def __init__(self):
        super(Head, self).__init__(
            linear=L.Linear(5, 2)
        )

    def __call__(self, _x, _is_train):
        y = self.linear(_x)
        return y

agent = Agent(Shared, Head, DemoEnv(), Replay())
train = Train(agent)
train.run()
