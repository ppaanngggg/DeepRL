import argparse
import logging
import typing

import gym
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from DeepRL.Agent import DDPGAgent
from DeepRL.Env import EnvAbstract, EnvState
from DeepRL.Replay import NaiveReplay
from DeepRL.Train import Train

logger = logging.getLogger()
logger.setLevel(logging.INFO)


class DemoEnv(EnvAbstract):
    def __init__(self):
        super().__init__()
        self.g = gym.make('Pendulum-v0')
        self.o: np.ndarray = None
        self.total_reward = 0.0
        self.render = False

    def startNewGame(self):
        self.o = self.g.reset()
        logger.info('total_reward: {}'.format(self.total_reward))
        if not self.render and 0. > self.total_reward > -100.:
            self.render = True
        self.total_reward = 0.0
        self.in_game = True

    def getState(self) -> EnvState:
        return EnvState(self.in_game, self.o)

    def doAction(self, _action: np.ndarray) -> float:
        self.o, reward, is_quit, _ = self.g.step(_action)
        self.in_game = not is_quit
        self.total_reward += reward
        if self.render:
            self.g.render()
        return reward / 10.

    def getInputs(
            self, _state_list: typing.Sequence[EnvState]
    ) -> np.ndarray:
        return np.array([
            d.state for d in _state_list
        ])


class ActorModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(3, 30)
        self.fc2 = nn.Linear(30, 1)

    def forward(self, x: Variable):
        hidden = F.relu(self.fc1(x))
        return F.tanh(self.fc2(hidden)) * 2.0


class CriticModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc_s = nn.Linear(3, 30)
        self.fc_a = nn.Linear(1, 30)
        self.fc_o = nn.Linear(30, 1)

    def forward(self, s: Variable, a: Variable):
        return self.fc_o(F.relu(self.fc_s(s) + self.fc_a(a)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()

    actor = ActorModel()
    critic = CriticModel()

    agent = DDPGAgent(
        _actor_model=actor, _critic_model=critic,
        _env=DemoEnv(), _gamma=0.9, _batch_size=32,
        _theta=0.15, _sigma=0.2, _update_rate=0.001,
        _replay=NaiveReplay(),
        _actor_optimizer=optim.Adam(actor.parameters(), lr=1e-4),
        _critic_optimizer=optim.Adam(critic.parameters(), lr=1e-3),
        _action_clip=2.0,
        _gpu=args.gpu
    )
    agent.config.epoch_show_log = 10000

    train = Train(
        agent,
        _epoch_max=10000,
        _step_init=1000,
        _step_train=1,
        _step_update_target=1,
        _step_save=100000000,
    )
    train.run()
