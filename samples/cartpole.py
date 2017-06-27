from DeepRL.Env import Env
from DeepRL.Agent import QAgent, NStepQAgent, AACAgent
from DeepRL.Replay import PrioritizedReplay
from DeepRL.Train import Train
from DeepRL.Test import Test

import gym
import tensorflow as tf
import numpy as np
import random


class DemoEnv(Env):

    def __init__(self):
        super(DemoEnv, self).__init__()
        self.g = gym.make('CartPole-v0')
        self.total_reward = 0.
        self.total_reward_list = []

    def doStartNewGame(self):
        self.total_reward_list.append(self.total_reward)
        print len(self.total_reward_list), self.total_reward_list[-1]
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
        # self.g.render()
        return reward

    def doGetX(self, _state):
        return _state.state['x']

    def doGetRandomAction(self, _state):
        return random.randint(0, 1)

    def doGetBestAction(self, _data, _state_list):
        return np.argmax(_data, 1).tolist()

    def doGetSoftAction(self, _data, _state_list):
        ret = []
        for d in _data:
            ret += np.random.choice(len(d), 1, p=d).tolist()
        return ret


def model(_x):
    w1 = tf.Variable(tf.random_uniform([4, 20]))
    b1 = tf.Variable(tf.zeros([20]))
    w2 = tf.Variable(tf.random_uniform([20, 2]))
    b2 = tf.Variable(tf.zeros([2]))

    hidden = tf.nn.relu(tf.matmul(_x, w1) + b1)
    output = tf.matmul(hidden, w2) + b2

    return output, [w1, b1, w2, b2]

agent = QAgent(
    _model=model, _env=DemoEnv(),
    _replay=PrioritizedReplay(),
    _optimizer=tf.train.RMSPropOptimizer(0.001, decay=0.99),
    _epsilon=1.0, _epsilon_decay=0.999, _epsilon_underline=0.1,
)
train = Train(agent, _step_save=1e8)
train.run()
