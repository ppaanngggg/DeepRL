import random
import typing
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim

from DeepRL.Agent.AgentAbstract import AgentAbstract, ACTION_TYPE
from DeepRL.Env import EnvAbstract, EnvState
from DeepRL.Replay.ReplayAbstract import ReplayAbstract, ReplayTuple


class DoubleDQNAgent(AgentAbstract):
    def __init__(
            self,
            _model: nn.Module,
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
        for param in self.target_q_func.parameters():
            param.requires_grad_(False)

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

    def updateEpsilon(self):
        self.config.epsilon = max(
            self.config.epsilon_underline,
            self.config.epsilon * self.config.epsilon_decay
        )

    def chooseAction(self, _state: EnvState) -> ACTION_TYPE:
        if self.config.is_train:
            # update epsilon
            self.updateEpsilon()
            random_value = random.random()
            if random_value < self.config.epsilon:
                # randomly choose
                return self.env.getRandomActions([_state])[0]

        # if eval or not use random action, use model to choose
        x = torch.Tensor(self.env.getInputs([_state]))
        with torch.no_grad():
            output = self.q_func(x).numpy()

        return self.env.getBestActions(output, [_state])[0]

    def doTrain(
            self, _batch_tuples: typing.Union[None, typing.Sequence[ReplayTuple]],
            _dataset=None
    ):
        # get inputs from batch
        prev_x = torch.Tensor(self.getPrevInputs(_batch_tuples))
        next_x = torch.Tensor(self.getNextInputs(_batch_tuples))

        # calc value estimate according to prev envs and prev actions
        prev_output = self.q_func(prev_x)
        prev_action = [d.action for d in _batch_tuples]
        prev_value = prev_output[range(len(prev_action)), prev_action]

        # calc target value estimate and loss
        with torch.no_grad():
            next_output = self.q_func(next_x)
        next_action = self.env.getBestActions(
            next_output.numpy(),
            [t.next_state for t in _batch_tuples]
        )
        # use target to re-estimate next value
        with torch.no_grad():
            next_output = self.target_q_func(next_x)
        next_value = next_output[range(len(next_action)), next_action]

        target_value = torch.Tensor([d.reward for d in _batch_tuples])
        target_value.add_(torch.Tensor([  # add gamma * next_value if game not ends
            d.next_state.in_game for d in _batch_tuples
        ]) * self.config.gamma * next_value)

        loss = self.criterion(
            prev_value, target_value
        )

        # update q func
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step(None)
