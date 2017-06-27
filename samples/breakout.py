from DeepRL.Env import Env
from DeepRL.Agent import QAgent, NStepQAgent
from DeepRL.Replay import TmpReplay, Replay, PrioritizedReplay
from DeepRL.Train import AsynTrain, Train
from DeepRL.Test import Test

import cv2
import tensorflow as tf
import gym
import numpy as np
import random

from time import time


class DemoEnv(Env):

    def __init__(self):
        super(DemoEnv, self).__init__()
        self.g = gym.make('Pong-v0')

    def doStartNewGame(self):
        o = self.g.reset()
        # self.g.render()
        o = 0.299 * o[:, :, 0] + 0.587 * o[:, :, 1] + 0.114 * o[:, :, 2]
        o /= 255.
        o = cv2.resize(o, (84, 84))
        self.buffer = [o] * 4

    def doGetState(self):
        x = np.stack(self.buffer, axis=-1)
        x = np.expand_dims(x.astype(np.float32), 0)
        return {'x': x}

    def doDoAction(self, _action):
        o, reward, quit, _ = self.g.step(_action)
        o = 0.299 * o[:, :, 0] + 0.587 * o[:, :, 1] + 0.114 * o[:, :, 2]
        o /= 255.
        o = cv2.resize(o, (84, 84))
        self.in_game = not quit
        # self.g.render()
        self.buffer = self.buffer[1:] + [o]
        return reward

    def doGetX(self, _state):
        return _state.state['x']

    def doGetRandomAction(self, _state):
        return random.randint(0, 5)

    def doGetBestAction(self, _data, _state_list):
        return np.argmax(_data, 1).tolist()


def model(_x):
    kernel1 = tf.Variable(tf.random_uniform(
        (8, 8, 4, 16), minval=-1, maxval=1))
    bias1 = tf.Variable(tf.zeros((16)))
    conv1 = tf.nn.relu(
        tf.nn.bias_add(
            tf.nn.conv2d(_x, kernel1, (1, 4, 4, 1), 'VALID'),
            bias1
        )
    )
    kernel2 = tf.Variable(tf.random_uniform(
        (4, 4, 16, 32), minval=-1, maxval=1))
    bias2 = tf.Variable(tf.zeros((32)))
    conv2 = tf.nn.relu(
        tf.nn.bias_add(
            tf.nn.conv2d(conv1, kernel2, (1, 2, 2, 1), 'VALID'),
            bias2
        )
    )
    reshape = tf.reshape(conv2, (-1, 2592))
    weight1 = tf.Variable(tf.random_uniform((2592, 256), minval=-1, maxval=1))
    bias3 = tf.Variable(tf.zeros((256)))
    hidden = tf.nn.relu(tf.matmul(reshape, weight1) + bias3)
    weight2 = tf.Variable(tf.random_uniform((256, 6), minval=-1, maxval=1))
    bias4 = tf.Variable(tf.zeros((6)))
    output = tf.matmul(hidden, weight2) + bias4

    return output, [kernel1, bias1, kernel2, bias2, weight1, bias3, weight2, bias4]


def create_agent():
    return QAgent(model, _env=DemoEnv(), _gpu=True,
                  _replay=TmpReplay(),
                  _epsilon=1.0, _epsilon_decay=0.999998, _epsilon_underline=0.1)

train = AsynTrain(create_agent,
                  _process_num=8,
                  _step_update_func=5,
                  _step_update_target=4e4,
                  _step_save=1e6,
                  _q_opt=tf.train.RMSPropOptimizer(0.00025, decay=0.99))
train.run()
