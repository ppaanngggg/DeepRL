import typing
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from DeepRL.Agent.AgentAbstract import AgentAbstract
from DeepRL.Env import EnvAbstract
from DeepRL.Replay.ReplayAbstract import ReplayAbstract, ReplayTuple


class QAgent(AgentAbstract):
    def __init__(
            self, _model: nn.Module,
            _env: EnvAbstract,
            _gamma: float, _batch_size: int,
            _epsilon_init: float,
            _epsilon_decay: float,
            _epsilon_underline: float,
            _replay: ReplayAbstract = None,
            _optimizer: optim.Optimizer = None,
            _err_clip: float = None, _grad_clip: float = None
    ):
        super().__init__(_env)

        self.q_func: nn.Module = _model
        self.target_q_func: nn.Module = deepcopy(_model)

        # set config
        self.config.gamma = _gamma
        self.config.batch_size = _batch_size
        self.config.epsilon = _epsilon_init
        self.config.epsilon_decay = _epsilon_decay
        self.config.epsilon_underline = _epsilon_underline
        self.config.err_clip = _err_clip
        self.config.grad_clip = _grad_clip

        self.replay = _replay

        self.criterion = nn.MSELoss()
        self.optimizer = _optimizer

    def func(
            self, _x_data: np.ndarray, _train: bool = True
    ) -> np.ndarray:
        x_var = Variable(torch.from_numpy(_x_data)).float()
        return self.q_func(x_var).data.numpy()

    def doTrain(self, _batch_tuples: typing.Sequence[ReplayTuple]):
        # get inputs from batch
        cur_x = self.getCurInputs(_batch_tuples)
        next_x = self.getNextInputs(_batch_tuples)
        cur_x = Variable(torch.from_numpy(cur_x)).float()
        next_x = Variable(torch.from_numpy(next_x)).float()

        cur_output = self.q_func(cur_x)
        cur_action = self.getActionData(
            cur_output.size(), [d.action for d in _batch_tuples]
        )
        cur_output = cur_output * Variable(torch.from_numpy(cur_action))
        cur_output = cur_output.sum(1)
        # compute forward
        next_output = self.q_func(next_x)
        next_action = self.env.getBestActions(
            next_output.data.numpy(),
            [t.next_state for t in _batch_tuples]
        )
        next_output = self.target_q_func(next_x)
        target_data = self.getQTargetData(
            next_output.data.numpy(), next_action, _batch_tuples
        )
        loss = self.criterion(
            cur_output, Variable(torch.from_numpy(target_data))
        )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
