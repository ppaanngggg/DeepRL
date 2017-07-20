import typing
from collections import deque

import numpy as np

from DeepRL.Env import EnvState
from DeepRL.Replay.ReplayAbstract import ReplayAbstract, ReplayTuple


class NaiveReplay(ReplayAbstract):
    def __init__(self, _size: int = 1e5):
        self.size = int(_size)
        self.memory_pool: typing.Deque[ReplayTuple] = deque(maxlen=self.size)

    def push(
            self, _state: EnvState,
            _action: int, _reward: float,
            _next_state: EnvState, _mask=None
    ):
        # store new tuples into tmp memory buffer
        self.memory_pool.append(
            ReplayTuple(_state, _action, _reward, _next_state)
        )

    def pull(
            self, _num: int
    ) -> typing.Sequence[ReplayTuple]:
        choices = []
        if len(self.memory_pool):
            choices = np.random.choice(
                len(self.memory_pool),
                min(len(self.memory_pool), _num),
                replace=False,
            )
        return [self.memory_pool[choice] for choice in choices]

    def merge(self):
        pass

    def __repr__(self) -> str:
        return '{}'.format(self.memory_pool)
