from DeepRL.Agent import *
from DeepRL.Env import Env
from DeepRL import Config
from DeepRL.Replay import *
from DeepRL.Train import Train
from DeepRL.Test import Test
import logging
import random
import numpy as np
import chainer.links as L
from chainer import optimizers
import gym

Config.step_save = 1000
Config.setp_update_target = 1000
Config.bootstrap = True
Config.double_q = True
Config.prioritized_replay = True
Config.gamma = 0.99
Config.epsilon = 0.5
Config.epsilon_decay = 0.995
Config.epsilon_underline = 0.01


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


class Shared(Chain):

    def __init__(self):
        super(Shared, self).__init__(
            linear=L.Linear(4, 20)
        )

    def __call__(self, _x, _is_train):
        y = F.relu(self.linear(_x))
        return y


class Head(Chain):

    def __init__(self):
        super(Head, self).__init__(
            linear=L.Linear(20, 2)
        )

    def __call__(self, _x, _is_train):
        y = self.linear(_x)
        return y

agent = Agent(Shared, Head, DemoEnv(),
              _optimizer=optimizers.RMSprop(), _replay=Replay())
train = Train(agent)
train.run()

# agent = Agent(Shared, Head, DemoEnv(), _pre_model='./models/step_6000')
# test = Test(agent)
# test.run()
