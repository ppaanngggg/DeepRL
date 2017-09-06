import argparse
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pendulum_ddpg import DemoEnv
from torch.autograd import Variable

from DeepRL.Agent import PPOAgent
from DeepRL.Replay import TmpReplay
from DeepRL.Train import Train

logger = logging.getLogger()
logger.setLevel(logging.INFO)


class PolicyModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(3, 30)
        self.fc2 = nn.Linear(30, 1)

        self.std = nn.Parameter(torch.zeros(1))

    def forward(self, x: Variable):
        hidden = F.relu(self.fc1(x))
        return F.tanh(self.fc2(hidden)) * 2.0, self.std


class ValueModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc_s = nn.Linear(3, 30)
        self.fc_o = nn.Linear(30, 1)

    def forward(self, s: Variable):
        return self.fc_o(F.relu(self.fc_s(s)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()

    policy_model = PolicyModel()
    value_model = ValueModel()

    agent = PPOAgent(
        _policy_model=policy_model,
        _value_model=value_model,
        _env=DemoEnv(),
        _replay=TmpReplay(),
        _policy_optimizer=optim.Adam(policy_model.parameters(), 1e-4),
        _value_optimizer=optim.Adam(value_model.parameters(), 1e-4),
        _action_clip=2.0,
        _gpu=args.gpu)
    agent.config.epoch_show_log = 10000

    train = Train(
        agent,
        _epoch_max=10000,
        _step_init=1000,
        _step_train=500,
        _step_update_target=1,
        _step_save=100000000, )
    train.run()
