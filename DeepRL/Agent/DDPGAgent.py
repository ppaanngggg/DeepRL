import typing
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from DeepRL.Agent.AgentAbstract import AgentAbstract
from DeepRL.Env import EnvAbstract, EnvState
from DeepRL.Replay.ReplayAbstract import ReplayAbstract, ReplayTuple


class DDPGAgent(AgentAbstract):
    def __init__(
            self,
            _actor_model: nn.Module,
            _critic_model: nn.Module,
            _env: EnvAbstract,
            _gamma: float = 0.9,
            _batch_size: int = 64,
            _theta: typing.Union[float, np.ndarray] = 0.1,
            _sigma: typing.Union[float, np.ndarray] = 0.1,
            _update_rate: float = 0.001,
            _replay: ReplayAbstract = None,
            _actor_optimizer: optim.Optimizer = None,
            _critic_optimizer: optim.Optimizer = None,
            _action_clip: float = 1.0,
            _gpu: bool = False
    ):
        super().__init__(_env)

        # set config
        self.config.gamma = _gamma
        self.config.batch_size = _batch_size
        self.config.action_clip = _action_clip
        self.config.gpu = _gpu

        self.p_func: nn.Module = _actor_model
        self.target_p_func: nn.Module = deepcopy(self.p_func)
        for param in self.target_p_func.parameters():
            param.requires_grad_(False)
        self.q_func: nn.Module = _critic_model
        self.target_q_func: nn.Module = deepcopy(self.q_func)
        for param in self.target_q_func.parameters():
            param.requires_grad_(False)
        # turn to gpu, if necessary
        if self.config.gpu:
            self.p_func.cuda()
            self.target_p_func.cuda()
            self.q_func.cuda()
            self.target_q_func.cuda()

        # ou explore rate
        self.theta = _theta
        self.sigma = _sigma
        self.explore_shift: np.ndarray = None

        # update target rate
        self.update_rate = _update_rate

        self.replay = _replay

        self.criterion = nn.MSELoss()

        self.actor_optim = _actor_optimizer
        self.critic_optim = _critic_optimizer

    def _explore_action(self, _action: np.ndarray):
        if self.explore_shift is None:  # init explore_shift
            self.explore_shift = np.zeros_like(_action)
        np.add(
            (1 - self.theta) * self.explore_shift,
            self.sigma * np.random.normal(
                size=self.explore_shift.shape
            ),
            out=self.explore_shift
        )
        np.add(
            _action, self.explore_shift, out=_action
        )
        np.clip(
            _action, -self.config.action_clip, self.config.action_clip,
            out=_action
        )

    def chooseAction(self, _state: EnvState) -> np.ndarray:
        x = torch.Tensor(self.env.getInputs([_state]))
        with torch.no_grad():
            output = self.p_func(x).numpy()[0]
        if self.config.is_train:
            self._explore_action(output)

        return output

    def doTrain(
            self,
            _batch_tuples: typing.Union[None, typing.Sequence[ReplayTuple]],
            _dataset=None
    ):
        # calc target value estimate and loss
        next_x = torch.Tensor(self.getNextInputs(_batch_tuples))
        with torch.no_grad():
            next_output = self.target_q_func(next_x, self.target_p_func(next_x))

        rewards = torch.Tensor([t.reward for t in _batch_tuples])
        rewards.unsqueeze_(1)
        target_value = next_output + self.config.gamma * rewards

        prev_x = torch.Tensor(self.getPrevInputs(_batch_tuples))
        prev_action = torch.Tensor([d.action for d in _batch_tuples])
        prev_output = self.q_func(prev_x, prev_action)
        critic_loss = self.criterion(prev_output, target_value)

        # update critic
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step(None)

        # estimate current value by current actor
        actor_loss = self.q_func(prev_x, self.p_func(prev_x))
        actor_loss = -actor_loss.mean()

        # update actor
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step(None)

    def updateTargetFunc(self):
        for tp, p in zip(
                self.target_p_func.parameters(),
                self.p_func.parameters(),
        ):
            tp.data = (1 - self.update_rate) * \
                      tp.data + self.update_rate * p.data
        for tp, p in zip(
                self.target_q_func.parameters(),
                self.q_func.parameters(), ):
            tp.data = (1 - self.update_rate) * \
                      tp.data + self.update_rate * p.data
