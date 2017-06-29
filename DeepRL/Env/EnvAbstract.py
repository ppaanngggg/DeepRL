import typing

import numpy as np

from DeepRL.Env.EnvState import EnvState


class EnvAbstract(object):
    def __init__(self):
        self.in_game = False

    def startNewGame(self):
        raise NotImplementedError

    def getState(self) -> EnvState:
        """
        :return: cur env state
        """
        raise NotImplementedError

    def doAction(self, _action: int) -> float:
        """
        step according to the action, and return reward

        :param _action:
        :return:
        """
        raise NotImplementedError

    def getInputs(
            self, _state_list: typing.Sequence[EnvState]
    ) -> np.ndarray:
        raise NotImplementedError

    def getRandomActions(
            self, _state_list: typing.Sequence[EnvState]
    ) -> typing.Sequence[int]:
        raise NotImplementedError

    def getBestActions(
            self, _data: np.ndarray,
            _state_list: typing.Sequence[EnvState]
    ) -> typing.Sequence[int]:
        raise NotImplementedError

    def getSoftActions(
            self, _data: np.ndarray,
            _state_list: typing.Sequence[EnvState]
    ) -> typing.Sequence[int]:
        raise NotImplementedError
