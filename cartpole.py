from DeepRL.Env import Env
from DeepRL.Agent import QAgent
from DeepRL.Replay import Replay
from DeepRL.Train import Train

from chainer import Chain, optimizers
import chainer.links as L
import chainer.functions as F

import gym
import numpy as np
import random


class DemoEnv(Env):

    def __init__(self):
        super(DemoEnv, self).__init__()
        self.g = gym.make('CartPole-v0')
        self.epoch = 0
        self.total_reward = 0.
        self.total_reward_list = []

    def doStartNewGame(self):
        self.epoch += 1
        self.total_reward_list.append(self.total_reward)
        print len(self.total_reward_list), self.total_reward_list[-1]
        if self.epoch == 300:
            np.save('pri', self.total_reward_list)
            raw_input()
        self.total_reward = 0.
        self.o = self.g.reset()

    def doGetState(self):
        return {'x': np.expand_dims(self.o.astype(np.float32), 0)}

    def doDoAction(self, _action):
        self.o, reward, quit, _ = self.g.step(_action)
        self.in_game = not quit
        self.total_reward += reward
        if self.total_reward == 200:
            self.in_game = False
        self.g.render()
        return reward

    def doGetX(self, _state):
        return _state.state['x']

    def doGetRandomAction(self, _state):
        return random.randint(0, 1)

    def doGetBestAction(self, _data, _state_list):
        return np.argmax(_data, 1).tolist()


class Mlp(Chain):

    def __init__(self):
        super(Mlp, self).__init__(
            l1=L.Linear(4, 20),
            l2=L.Linear(20, 2)
        )

    def __call__(self, _x):
        y = self.l1(_x)
        y = F.relu(y)
        y = self.l2(y)
        return y

agent = QAgent(_model=Mlp, _env=DemoEnv(),
               _optimizer=optimizers.Adam(),
               _replay=Replay())
train = Train(agent)
train.run()
