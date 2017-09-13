import math
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
            _gamma: float = 0.9,
            _tau: float = 0.95,
            _rate_clip: float = 0.2,
            _beta_entropy: float = 0.01,
            _batch_size: int = 64,
            _train_epoch: int = 10,
            _replay: ReplayAbstract = None,
            _policy_optimizer: optim.Optimizer = None,
            _value_optimizer: optim.Optimizer = None,
            _action_clip: float = 1.0,
            _gpu: bool = False, ):
        super().__init__(_env)

        self.config.gamma = _gamma
        self.config.tau = _tau  # gae
        self.config.rate_clip = _rate_clip  # clip for rate, 0.8 ~ 1.2 default
        self.config.beta_entropy = _beta_entropy
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
        x_var = self.np2var(x_data, True)

        self.p_func.eval()
        action_mean, action_log_std = self.p_func(x_var)
        action_mean = action_mean.data
        action_log_std = action_log_std.data

        if self.config.is_train:
            action_std = action_log_std.exp().expand_as(action_mean)
            random_action = torch.normal(action_mean, action_std)
            random_action.clamp_(
                -self.config.action_clip, self.config.action_clip
            )
            if self.config.gpu:
                random_action = random_action.cpu()
            return random_action.numpy()[0]
        else:
            return action_mean.numpy()[0]

    def np2var(self, _np_arr: np.ndarray, _volatile: bool = False) -> Variable:
        tmp = torch.from_numpy(_np_arr)
        if self.config.gpu:
            tmp = tmp.cuda()
        return Variable(tmp, volatile=_volatile)

    def trainValueModel(self, _x, _target):
        x_var = Variable(_x)
        target_var = Variable(_target)

        self.value_optim.zero_grad()

        self.v_func.train()
        output = self.v_func(x_var)
        loss = self.criterion(output, target_var)
        loss.backward()

        self.value_optim.step()

    @staticmethod
    def getLogProb(_action: Variable, _mean: Variable, _log_std: Variable):
        std = torch.exp(_log_std)
        var = torch.pow(std, 2)
        log_prob = -torch.pow(_action - _mean, 2) / (2.0 * var)
        log_prob = log_prob - 0.5 * math.log(2 * math.pi) - _log_std
        return log_prob.sum(1)

    @staticmethod
    def getEntropy(_log_std: Variable):
        term = 0.5 * math.log(2 * math.pi * math.e)
        return term + _log_std

    def trainPolicyModel(
            self, _status, _action, _advantage
    ):
        status_var = Variable(_status)
        action_var = Variable(_action)
        adv_var = Variable(_advantage)

        self.policy_optim.zero_grad()

        self.p_func.train()
        self.target_p_func.train()
        new_mean, new_log_std = self.p_func(status_var)
        old_mean, old_log_std = self.target_p_func(status_var)

        new_log_prob = self.getLogProb(action_var, new_mean, new_log_std)
        old_log_prob = self.getLogProb(action_var, old_mean, old_log_std)

        entropy = self.getEntropy(new_log_std)

        rate = torch.exp(new_log_prob - old_log_prob)  # real prob rate
        rate_clip = torch.clamp(
            rate, 1 - self.config.rate_clip, 1 + self.config.rate_clip)
        final_rate = torch.min(rate, rate_clip)
        loss = -torch.mean(final_rate * adv_var) - \
            self.config.beta_entropy * entropy.mean()

        loss.backward()

        self.policy_optim.step()

    def getValues(self, _x: np.ndarray) -> np.ndarray:
        x_var = self.np2var(_x, _volatile=True)
        values = self.v_func(x_var).data
        if self.config.gpu:
            values = values.cpu()
        values.squeeze_()
        return values.numpy()

    def getDataset(self, _batch_tuples: typing.List[ReplayTuple]):
        status_arr = self.getPrevInputs(_batch_tuples)
        action_arr = np.array([d.action for d in _batch_tuples])

        # get value model's value of status
        value_arr = self.getValues(status_arr)
        # alloc space for true returns and advantages
        return_arr = np.zeros(len(_batch_tuples), dtype=np.float32)
        adv_arr = np.zeros(len(_batch_tuples), dtype=np.float32)

        prev_value = 0.0
        prev_return = 0.0
        prev_advantage = 0.0
        for i in reversed(range(len(_batch_tuples))):  # iter dec
            batch_tuple = _batch_tuples.pop()
            if batch_tuple.next_state.in_game:  # if game still continues
                return_arr[i] = batch_tuple.reward + \
                    self.config.gamma * prev_return
                delta = batch_tuple.reward + \
                    self.config.gamma * prev_value - value_arr[i]
                adv_arr[i] = delta + self.config.gamma * \
                    self.config.tau * prev_advantage
            else:  # if game ends
                return_arr[i] = batch_tuple.reward
                delta = batch_tuple.reward - value_arr[i]
                adv_arr[i] = delta
            prev_value = value_arr[i]
            prev_return = return_arr[i]
            prev_advantage = adv_arr[i]

        return status_arr, action_arr, return_arr, adv_arr

    def doTrain(
            self, _batch_tuples: typing.Union[None, typing.List[ReplayTuple]],
            _dataset=None
    ):
        if _dataset is not None:
            status_arr, action_arr, return_arr, adv_arr = _dataset
        elif _batch_tuples is not None:
            status_arr, action_arr, return_arr, adv_arr = self.getDataset(
                _batch_tuples)
        else:
            raise Exception('_batch_tuples and _dataset are both None')

        status_arr = torch.from_numpy(status_arr)
        action_arr = torch.from_numpy(action_arr)
        return_arr = torch.from_numpy(return_arr)
        adv_arr = torch.from_numpy(adv_arr)
        if self.config.gpu:
            status_arr = status_arr.cuda()
            action_arr = action_arr.cuda()
            return_arr = return_arr.cuda()
            adv_arr = adv_arr.cuda()
        advantage_arr = (adv_arr - adv_arr.mean()) / adv_arr.std()

        for _ in range(self.config.train_epoch):  # train several epochs
            rand_idx = torch.from_numpy(np.random.permutation(len(status_arr)))
            if self.config.gpu:
                rand_idx = rand_idx.cuda()
            for begin_idx in range(0, len(rand_idx), self.config.batch_size):
                end_idx = begin_idx + self.config.batch_size
                batch_idx = rand_idx[begin_idx:end_idx]
                batch_status_arr = status_arr[batch_idx]
                batch_action_arr = action_arr[batch_idx]
                batch_return_arr = return_arr[batch_idx]
                batch_advantage_arr = advantage_arr[batch_idx]

                self.trainValueModel(
                    _x=batch_status_arr,
                    _target=batch_return_arr
                )
                self.trainPolicyModel(
                    _status=batch_status_arr,
                    _action=batch_action_arr,
                    _advantage=batch_advantage_arr
                )

        # update old policy model
        self.target_p_func.load_state_dict(self.p_func.state_dict())

    def updateTargetFunc(self):
        pass
