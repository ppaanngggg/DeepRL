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


class DDPGAgent(AgentAbstract):
    def __init__(
            self,
            _actor_model: nn.Module,
            _critic_model: nn.Module,
            _env: EnvAbstract,
            _gamma: float = 0.9, _batch_size: int = 64,
            _theta: typing.Union[float, np.ndarray] = 0.1,
            _sigma: typing.Union[float, np.ndarray] = 0.1,
            _update_rate: float = 0.001,
            _replay: ReplayAbstract = None,
            _actor_optimizer: optim.Optimizer = None,
            _critic_optimizer: optim.Optimizer = None,
            _action_clip: float = 1.0,
            _gpu: bool = False,
    ):
        super().__init__(_env)

        # set config
        self.config.gamma = _gamma
        self.config.batch_size = _batch_size
        self.config.action_clip = _action_clip
        self.config.gpu = _gpu

        self.p_func: nn.Module = _actor_model
        self.target_p_func: nn.Module = deepcopy(self.p_func)
        for p in self.target_p_func.parameters():
            p.requires_grad = False
        self.q_func: nn.Module = _critic_model
        self.target_q_func: nn.Module = deepcopy(self.q_func)
        for p in self.target_q_func.parameters():
            p.requires_grad = False
        # turn to gpu, if necessary
        if self.config.gpu:
            self.p_func.cuda()
            self.target_p_func.cuda()
            self.q_func.cuda()
            self.target_q_func.cuda()

        # explore rate
        self.theta = _theta
        self.sigma = _sigma
        self.current_x: np.ndarray = None

        # update target rate
        self.update_rate = _update_rate

        self.replay = _replay

        self.criterion = nn.MSELoss()

        self.actor_optim = _actor_optimizer
        self.critic_optim = _critic_optimizer

    def _action_random(self, _action: np.ndarray) -> np.ndarray:
        if self.current_x is None:
            self.current_x = np.zeros_like(_action)
        diff_x = self.theta * -self.current_x
        diff_x = diff_x + self.sigma * \
            np.random.normal(size=self.current_x.shape)
        self.current_x = self.current_x + diff_x
        tmp_action = _action + self.current_x
        return tmp_action

    def chooseAction(self, _state: EnvState) -> np.ndarray:
        x_data = self.env.getInputs([_state])
        a = self.func(x_data, False)[0]
        if self.config.is_train:
            return np.clip(
                self._action_random(a),
                -self.config.action_clip,
                self.config.action_clip,
            )
        else:
            return a

    def func(
            self, _x_data: np.ndarray, _train: bool = True
    ) -> np.ndarray:
        x_data = torch.from_numpy(_x_data).float()
        if self.config.gpu:
            x_data = x_data.cuda()
        x_var = Variable(
            x_data, volatile=not _train
        )
        output = self.p_func(x_var).data
        if self.config.gpu:
            output = output.cpu()
        return output.numpy()

    def doTrain(self, _batch_tuples: typing.Sequence[ReplayTuple]):
        prev_x = torch.from_numpy(self.getPrevInputs(_batch_tuples)).float()
        next_x = torch.from_numpy(self.getNextInputs(_batch_tuples)).float()
        prev_action = torch.from_numpy(
            np.array([d.action for d in _batch_tuples])).float()
        if self.config.gpu:
            prev_x = prev_x.cuda()
            next_x = next_x.cuda()
            prev_action = prev_action.cuda()
        prev_x = Variable(prev_x)
        next_x = Variable(next_x, volatile=True)
        prev_action = Variable(prev_action)

        # calc target value estimate and loss
        next_action = self.target_p_func(next_x)
        next_output = self.target_q_func(next_x, next_action).data
        if self.config.gpu:
            next_output = next_output.cpu()
        target_value = self.getQTargetData(
            next_output.numpy(), next_action, _batch_tuples
        )
        target_value = torch.from_numpy(target_value).float()
        if self.config.gpu:
            target_value = target_value.cuda()
        prev_output = self.q_func(prev_x, prev_action)
        critic_loss = self.criterion(
            prev_output, Variable(target_value)
        )

        # update critic
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # estimate current value by current actor
        actor_loss = self.q_func(prev_x, self.p_func(prev_x))
        actor_loss = -actor_loss.mean()

        # update actor
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

    def getQTargetData(
            self, _next_output: np.ndarray,
            _next_action: typing.Sequence[int],
            _batch_tuples: typing.Sequence[ReplayTuple]
    ) -> np.ndarray:
        reward_list = [t.reward for t in _batch_tuples]
        reward_arr = np.expand_dims(np.array(reward_list), 1)
        return reward_arr + self.config.gamma * _next_output

    def updateTargetFunc(self):
        for tp, p in zip(
                self.target_p_func.parameters(),
                self.p_func.parameters(),
        ):
            tp.data = (1 - self.update_rate) * \
                tp.data + self.update_rate * p.data
        for tp, p in zip(
                self.target_q_func.parameters(),
                self.q_func.parameters(),
        ):
            tp.data = (1 - self.update_rate) * \
                tp.data + self.update_rate * p.data
