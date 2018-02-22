import argparse
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from DeepRL.Agent import PPOAgent
from DeepRL.Replay import TmpReplay
from DeepRL.Train import AsynTrainEpoch, TrainEpoch
from envs.pendulum_env import DemoEnv

logger = logging.getLogger()
logger.setLevel(logging.INFO)


class PolicyModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(3, 30)
        self.fc2 = nn.Linear(30, 1)

        self.log_std = nn.Parameter(torch.zeros(1))

    def forward(self, x: Variable):
        hidden = F.relu(self.fc1(x))
        return F.tanh(self.fc2(hidden)) * 2.0, self.log_std


class ValueModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc_s = nn.Linear(3, 30)
        self.fc_o = nn.Linear(30, 1)

    def forward(self, s: Variable):
        return self.fc_o(F.relu(self.fc_s(s)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--asyn', action='store_true')
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()

    policy_model = PolicyModel()
    value_model = ValueModel()
    env = DemoEnv()

    agent = PPOAgent(
        _policy_model=policy_model,
        _value_model=value_model,
        _env=env,
        _beta_entropy=0.01,
        _replay=TmpReplay(),
        _policy_optimizer=optim.Adam(policy_model.parameters(), 1e-4),
        _value_optimizer=optim.Adam(value_model.parameters(), 1e-4),
        _action_clip=2.0,
        _gpu=args.gpu
    )
    agent.config.epoch_show_log = 10000

    if args.asyn:
        train = AsynTrainEpoch(
            _agent=agent,
            _env=env,
            _epoch_max=1000,
            _epoch_train=5,
            _train_update_target=100,
            _train_save=100000,
            _process_core=5
        )
        train.run()
    else:
        train = TrainEpoch(
            _agent=agent,
            _env=env,
            _epoch_max=1000,
            _epoch_train=5,
            _epoch_update_target=100,
            _epoch_save=100000,
        )
        train.run()
