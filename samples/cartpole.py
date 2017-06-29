import random
import typing

import gym
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from DeepRL.Agent import QAgent
from DeepRL.Env import EnvState, EnvAbstract
from DeepRL.Replay import NaiveReplay
from DeepRL.Train import Train


class DemoEnv(EnvAbstract):
    def __init__(self):
        super().__init__()
        self.g = gym.make('CartPole-v0')
        self.o: np.ndarray = None
        self.total_reward = 0.

    def startNewGame(self):
        self.o = self.g.reset()
        self.total_reward = 0.
        self.in_game = True

    def getState(self) -> EnvState:
        return EnvState(self.in_game, {'x': self.o})

    def doAction(self, _action: int) -> float:
        self.o, reward, is_quit, _ = self.g.step(_action)
        self.in_game = not is_quit
        self.total_reward += reward
        if self.total_reward == 200:
            self.in_game = False
        self.g.render()
        return reward

    def getInput(
            self, _state_list: typing.Sequence[EnvState]
    ) -> np.ndarray:
        return np.array([
            d.state['x'] for d in _state_list
        ])

    def getRandomAction(
            self, _state_list: typing.Sequence[EnvState]
    ) -> typing.Sequence[int]:
        return [random.randint(0, 1) for _ in _state_list]

    def getBestAction(
            self, _data: np.ndarray,
            _state_list: typing.Sequence[EnvState]
    ) -> typing.Sequence[int]:
        return np.argmax(_data, 1)

    def getSoftAction(
            self, _data: np.ndarray,
            _state_list: typing.Sequence[EnvState]
    ) -> typing.Sequence[int]:
        ret = []
        for d in _data:
            ret += np.random.choice(len(d), 1, p=d).tolist()
        return ret


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(4, 4)
        self.fc2 = nn.Linear(4, 2)

    def forward(self, x: Variable):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


if __name__ == '__main__':
    model = Model()
    agent = QAgent(
        _model=model, _env=DemoEnv(),
        _gamma=0.9, _batch_size=32,
        _epsilon_init=1.0, _epsilon_decay=0.9999,
        _epsilon_underline=0.05,
        _replay=NaiveReplay(),
        _optimizer=optim.SGD(model.parameters(), 0.001, 0.9)
    )
    train = Train(
        agent,
        _epoch_max=10000,
        _step_init=100,
        _step_train=1,
        _step_update_target=1000,
        _step_save=10000000,
    )
    train.run()
