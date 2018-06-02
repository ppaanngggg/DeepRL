import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from DeepRL.Agent import DoubleDQNAgent
from DeepRL.Env.gym_wrapper import CartPoleEnv
from DeepRL.Replay import NaiveReplay
from DeepRL.Train import Train

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.INFO)


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(4, 4)
        self.fc2 = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor):
        x = F.tanh(self.fc1(x))
        x = self.fc2(x)
        return x


if __name__ == '__main__':
    model = Model()
    agent = DoubleDQNAgent(
        _model=model, _env=CartPoleEnv(),
        _gamma=0.9, _batch_size=32,
        _epsilon_init=1.0, _epsilon_decay=0.9999,
        _epsilon_underline=0.1,
        _replay=NaiveReplay(),
        _optimizer=optim.SGD(model.parameters(), 0.001, 0.9)
    )
    agent.config.epoch_show_log = 100
    train = Train(
        agent,
        _epoch_max=10000,
        _step_init=100,
        _step_train=1,
        _step_update_target=1000,
        _step_save=10000000,
    )
    train.run()
