import typing

import numpy as np

from DeepRL.Env import EnvState


class ReplayTuple:
    def __init__(
            self, _state: EnvState,
            _action: typing.Union[int, np.ndarray],
            _reward: float, _next_state: EnvState
    ):
        self.state = _state
        self.action = _action
        self.reward = _reward
        self.next_state = _next_state

    def __repr__(self) -> str:
        tmp = \
            "\n----- begin -----\n" \
            "-- state: \n{}\n" \
            "-- action: {}\n" \
            "-- reward: {}\n" \
            "-- next_state: \n{}\n" \
            "----- end -----\n"
        return tmp.format(
            self.state, self.action, self.reward, self.next_state
        )


class ReplayAbstract:
    def push(
            self, _state: EnvState,
            _action: typing.Union[int, np.ndarray],
            _reward: float, _next_state: EnvState,
    ):
        raise NotImplementedError()

    def pull(self, _num: int) -> typing.Sequence[ReplayTuple]:
        raise NotImplementedError()

    def merge(self):
        raise NotImplementedError()
