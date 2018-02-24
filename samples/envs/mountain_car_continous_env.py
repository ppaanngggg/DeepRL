import logging
import typing

import gym
import numpy as np

from DeepRL.Env import EnvAbstract, EnvState

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger()
logger.setLevel(logging.INFO)


class MountainCarContinuousEnv(EnvAbstract):
    def __init__(self):
        super().__init__()
        self.g = gym.make('MountainCarContinuous-v0')
        self.o: np.ndarray = None
        self.total_step = 0
        self.total_reward = 0.0
        self.render = False

    def startNewGame(self):
        self.o = self.g.reset()
        self.o = self.o.astype(np.float32)
        self.total_reward = 0.0
        self.in_game = True

    def getState(self) -> EnvState:
        return EnvState(self.in_game, self.o)

    def doAction(self, _action: np.ndarray) -> float:
        self.o, reward, is_quit, _ = self.g.step(_action)
        self.o = self.o.astype(np.float32)
        self.in_game = not is_quit
        self.total_reward += reward
        if not self.in_game:
            logger.info('total_reward: {}'.format(self.total_reward))
            # if not self.render and self.total_reward > 90.0:
            #     self.render = True
        if self.render:
            self.g.render()
        return min(reward, 1.0)

    def getInputs(self, _state_list: typing.Sequence[EnvState]) -> np.ndarray:
        return np.array([d.state for d in _state_list])

    def getRandomActions(
            self, _state_list: typing.Sequence[EnvState]
    ) -> typing.Sequence[int]:
        pass

    def getBestActions(
            self, _data: np.ndarray, _state_list: typing.Sequence[EnvState]
    ) -> typing.Sequence[int]:
        pass

    def getSoftActions(
            self, _data: np.ndarray, _state_list: typing.Sequence[EnvState]
    ) -> typing.Sequence[int]:
        pass
