import random
import typing

import gym
import numpy as np

from DeepRL.Env import EnvState, EnvAbstract


class CartPoleEnv(EnvAbstract):

    def __init__(self):
        super().__init__()
        self.gym_env = gym.make('CartPole-v0')
        self.status: np.ndarray = None
        self.total_reward = 0.
        self.render = False

    def startNewGame(self):
        self.status = self.gym_env.reset()
        if not self.render and self.total_reward > 195:
            self.render = True
        self.total_reward = 0.
        self.in_game = True

    def getState(self) -> EnvState:
        return EnvState(self.in_game, self.status)

    def doAction(self, _action: int) -> float:
        self.status, reward, is_quit, _ = self.gym_env.step(_action)
        self.in_game = not is_quit
        self.total_reward += reward
        if self.render:
            self.gym_env.render()
        return reward

    def getInputs(
            self, _state_list: typing.Sequence[EnvState]
    ) -> np.ndarray:
        return np.stack([
            d.state for d in _state_list
        ])

    def getRandomActions(
            self, _state_list: typing.Sequence[EnvState]
    ) -> typing.Sequence[int]:
        return [random.randint(0, 1) for _ in _state_list]

    def getBestActions(
            self, _actions: np.ndarray,
            _state_list: typing.Sequence[EnvState]
    ) -> typing.Sequence[int]:
        return np.argmax(_actions, 1)

    def getSoftActions(
            self, _actions: np.ndarray,
            _state_list: typing.Sequence[EnvState]
    ) -> typing.Sequence[int]:
        pass
