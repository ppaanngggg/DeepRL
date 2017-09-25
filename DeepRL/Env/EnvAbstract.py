import typing

import numpy as np

from DeepRL.Env.EnvState import EnvState


class EnvAbstract(object):
    def __init__(self):
        self.in_game = False
        self.total_reward = None

    def startNewGame(self):
        """
        reset and start a new game,
        !!! you have to create the init state !!!
        !!! you have to set in_game to True if finished !!!
        :return: none
        """
        raise NotImplementedError

    def getState(self) -> EnvState:
        """
        :return: cur env state
        """
        raise NotImplementedError

    def doAction(self, _action: typing.Union[int, np.ndarray]) -> float:
        """
        step according to the action, and return reward

        :param _action: action
        :return: reward
        """
        raise NotImplementedError

    def getInputs(
            self, _state_list: typing.Sequence[EnvState]
    ) -> np.ndarray:
        """
        get the inputs from states to model

        :param _state_list: list of states
        :return: array of input
        """
        raise NotImplementedError

    def getRandomActions(
            self, _state_list: typing.Sequence[EnvState]
    ) -> typing.Sequence[int]:
        """
        return the random actions according to states

        :param _state_list:
        :return:
        """
        raise NotImplementedError

    def getBestActions(
            self, _data: np.ndarray,
            _state_list: typing.Sequence[EnvState]
    ) -> typing.Sequence[int]:
        """
        return best actions according to model's output and states

        :param _data:
        :param _state_list:
        :return:
        """
        raise NotImplementedError

    def getSoftActions(
            self, _data: np.ndarray,
            _state_list: typing.Sequence[EnvState]
    ) -> typing.Sequence[int]:
        """
        return soft actions according to model's output and states

        :param _data:
        :param _state_list:
        :return:
        """
        raise NotImplementedError
