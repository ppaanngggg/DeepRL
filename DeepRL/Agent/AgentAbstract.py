import logging
import random
import typing

import numpy as np
import torch
import torch.nn as nn

from DeepRL.Env import EnvAbstract, EnvState
from DeepRL.Replay.ReplayAbstract import ReplayAbstract, ReplayTuple

logger = logging.getLogger()
logger.setLevel(logging.INFO)


class Config(object):
    """
    Config for Agent
    """

    def __init__(self):
        """
        Alloc all configs
        """
        self.gpu = False

        self.is_train = True

        self.gamma = None  # gamma, decay param of reward
        self.batch_size = None  # batch size of train

        # randomly choose action
        self.epsilon = None
        self.epsilon_decay = None
        self.epsilon_underline = None

        self.err_clip = None  # whether to clip err
        self.grad_clip = None  # whether to clip grad
        self.action_clip = None  # whether to clip action

        # bootstrapped, num of heads
        self.K = None
        # bootstrapped, p of mask
        self.mask_p = None

        # NFSP when to use p_func
        self.eta = None

        # for N step
        self.step_len = None
        # beta of entropy
        self.beta_entropy = None

        self.epoch_show_log = 1  # when to show log


class AgentAbstract:
    """
    base class of other agents
    """

    def __init__(self, _env: EnvAbstract):
        # alloc all models
        self.v_func: nn.Module = None
        self.target_v_func: nn.Module = None

        self.q_func: nn.Module = None
        self.target_q_func: nn.Module = None

        self.p_func: nn.Module = None
        self.target_p_func: nn.Module = None

        # alloc env
        self.env = _env

        # set config, and set device
        self.config = Config()

        # init epoch
        self.epoch = 0

        self.replay: ReplayAbstract = None

    def training(self):
        """
        set agent to train mode
        """
        self.config.is_train = True

    def evaluating(self):
        """
        set agent to evaluate mode
        """
        self.config.is_train = False

    def startNewGame(self):
        """
        normal start new game, suitable for most agent
        """
        while not self.env.in_game:
            self.env.startNewGame()
        self.epoch += 1

    def step(self) -> bool:
        """
        agent will get cur state and choose one action and execute

        Returns:
            still in game or not
        """

        # get current state
        cur_state = self.env.getState()
        # choose action in step
        action = self.chooseAction(cur_state)
        # do action and get reward
        reward = self.env.doAction(action)

        if self.epoch % self.config.epoch_show_log == 0:
            logger.info('Action: {}; Reward: {}'.format(action, reward))

        if self.config.is_train:
            # get new state
            next_state = self.env.getState()
            # store replay_tuple into memory pool
            self.replay.push(cur_state, action, reward, next_state)

        return self.env.in_game

    def chooseAction(self, _state: EnvState) -> int:
        """
        choose action by special model in special state,
        suitable for most agent
        """
        if self.config.is_train:
            # update epsilon
            self.updateEpsilon()
            random_value = random.random()
            if random_value < self.config.epsilon:
                # randomly choose
                return self.env.getRandomActions([_state])[0]
            else:
                # use model to choose
                x_data = self.env.getInputs([_state])
                output = self.func(x_data, False)

                if self.epoch % self.config.epoch_show_log == 0:
                    logger.info(str(output))
                return self.env.getBestActions(output, [_state])[0]
        else:
            x_data = self.env.getInputs([_state])
            output = self.func(x_data, False)
            logger.info(output)
            return self.env.getBestActions(output, [_state])[0]

    def updateEpsilon(self):
        self.config.epsilon = max(
            self.config.epsilon_underline,
            self.config.epsilon * self.config.epsilon_decay
        )

    def func(
            self, _x_data: np.ndarray, _train: bool = True
    ) -> np.ndarray:
        """
        get the output of model

        :param _x_data: input
        :param _train: this output is for training or not
        :return: output of model
        """
        raise NotImplementedError

    def train(self):
        """
        train model
        """
        # pull tuples from memory pool
        batch_tuples = self.replay.pull(self.config.batch_size)
        if not len(batch_tuples):
            return

        # train and
        self.doTrain(batch_tuples)

        # set err and merge
        self.replay.merge()

    def doTrain(self, _batch_tuples: typing.Sequence[ReplayTuple]):
        """
        do train detail, need to be overwritten
        """
        raise NotImplementedError

    def getPrevInputs(self, _batch_tuples: typing.Sequence[ReplayTuple]):
        """
        get and stack cur inputs from tuples
        """
        return self.env.getInputs([t.state for t in _batch_tuples])

    def getNextInputs(self, _batch_tuples: typing.Sequence[ReplayTuple]):
        """
        get and stack next inputs from tuples
        """
        return self.env.getInputs([t.next_state for t in _batch_tuples])

    def getActionData(
            self, _shape: typing.Tuple[int],
            _actions: typing.Sequence[int],
    ) -> np.ndarray:
        action_data = np.zeros(_shape, dtype=np.float32)
        for i in range(len(_actions)):
            action_data[i, _actions[i]] = 1.
        return action_data

    def getQTargetData(
            self, _next_output: np.ndarray,
            _next_action: typing.Sequence[int],
            _batch_tuples: typing.Sequence[ReplayTuple]
    ) -> np.ndarray:
        target_data = np.zeros(len(_batch_tuples), np.float32)
        for i in range(len(_batch_tuples)):
            target_value = _batch_tuples[i].reward
            # if not terminal state
            if _batch_tuples[i].next_state.in_game:
                target_value += \
                    self.config.gamma * _next_output[i][_next_action[i]]
            target_data[i] = target_value
        return target_data

    @staticmethod
    def _update_target_func(_target_func: nn.Module, _func: nn.Module):
        if _target_func is not None:
            assert _func is not None
            _target_func.load_state_dict(_func.state_dict())

    def updateTargetFunc(self):
        """
        update target funcs if exist
        """
        logger.info('update target func')
        AgentAbstract._update_target_func(self.target_v_func, self.v_func)
        AgentAbstract._update_target_func(self.target_q_func, self.q_func)
        AgentAbstract._update_target_func(self.target_p_func, self.p_func)

    def save(self, _epoch, _step):
        if self.p_func is not None:
            torch.save(
                self.p_func.state_dict(),
                './save/p_{}_{}'.format(_epoch, _step)
            )
        if self.q_func is not None:
            torch.save(
                self.q_func.state_dict(),
                './save/q_{}_{}'.format(_epoch, _step)
            )
        if self.v_func is not None:
            torch.save(
                self.v_func.state_dict(),
                './save/v_{}_{}'.format(_epoch, _step)
            )
