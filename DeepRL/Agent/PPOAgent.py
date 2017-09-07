import typing
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from DeepRL.Agent.AgentAbstract import AgentAbstract
from DeepRL.Env import EnvAbstract, EnvState
from DeepRL.Replay.ReplayAbstract import ReplayAbstract, ReplayTuple


class PPOAgent(AgentAbstract):
    def __init__(
            self,
            _policy_model: nn.Module,
            _value_model: nn.Module,
            _env: EnvAbstract,
            _gamma: float=0.9,
            _tau: float=0.97,
            _batch_size: int=64,
            _train_epoch: int=10,
            _replay: ReplayAbstract=None,
            _policy_optimizer: optim.Optimizer=None,
            _value_optimizer: optim.Optimizer=None,
            _action_clip: float=1.0,
            _gpu: bool=False, ):
        super().__init__(_env)

        self.config.gamma = _gamma
        self.config.tau = _tau  # gae
        self.config.batch_size = _batch_size  # mini batch in a train callback
        self.config.train_epoch = _train_epoch  # epoch in a train callback
        self.config.action_clip = _action_clip
        self.config.gpu = _gpu

        self.p_func: nn.Module = _policy_model
        self.target_p_func: nn.Module = deepcopy(self.p_func)
        for p in self.target_p_func.parameters():
            p.requires_grad = False
        self.v_func: nn.Module = _value_model

        if self.config.gpu:
            self.p_func.cuda()
            self.target_p_func.cuda()
            self.v_func.cuda()

        self.replay = _replay

        self.criterion = nn.MSELoss()

        self.policy_optim = _policy_optimizer
        self.value_optim = _value_optimizer

    def chooseAction(self, _state: EnvState) -> np.ndarray:
        x_data = self.env.getInputs([_state])
        x_data = torch.from_numpy(x_data).float()
        if self.config.gpu:
            x_data = x_data.cuda()
        x_var = Variable(x_data, volatile=True)
        action_mean, action_std = self.p_func(x_var)
        if self.config.gpu:
            action_mean = action_mean.cpu()
            action_std = action_std.cpu()

        if self.config.is_train:
            random_action = np.random.normal(
                action_mean.data.numpy()[0],
                action_std.data.numpy(), )
            np.clip(
                random_action,
                -self.config.action_clip,
                self.config.action_clip,
                out=random_action, )
            return random_action
        else:
            return action_mean.data.numpy()[0]

    def doTrain(self, _batch_tuples: typing.Sequence[ReplayTuple]):
        prev_x = torch.from_numpy(self.getPrevInputs(_batch_tuples)).float()
        next_x = torch.from_numpy(self.getNextInputs(_batch_tuples)).float()
        prev_action = torch.from_numpy(np.array([d.action for d in _batch_tuples])).float()

        for _ in range(self.config.train_epoch):
            if self.config.gpu:
                prev_x = prev_x.cuda()
            prev_x_var = Variable(prev_x, volatile=True)
            values = self.v_func(prev_x_var).data
            if self.config.gpu:
                values = values.cpu()
            returns = torch.Tensor(len(_batch_tuples))
            advantages = torch.Tensor(len(_batch_tuples))

            prev_value = 0.0
            prev_return = 0.0
            prev_advantage = 0.0
            for i in reversed(range(_batch_tuples)):



        # update old policy model
        self.target_p_func.load_state_dict(self.p_func.state_dict())

    def updateTargetFunc(self):
        pass
