import typing

import numpy as np
from DeepRL.Env import EnvState
from DeepRL.Replay.ReplayAbstract import ReplayAbstract, ReplayTuple


class NaiveReplay(ReplayAbstract):
    def __init__(self, _N: int = 1e4):
        self.N = int(_N)
        self.tmp_memory_pool: typing.List[ReplayTuple] = []
        self.memory_pool: typing.List[ReplayTuple] = []

    def push(
            self, _state: EnvState,
            _action: int, _reward: float,
            _next_state: EnvState, _mask=None
    ):
        # store new tuples into tmp memory buffer
        self.tmp_memory_pool.append(
            ReplayTuple(_state, _action, _reward, _next_state)
        )

    def pull(
            self, _num: int
    ) -> typing.Sequence[ReplayTuple]:
        choices = []
        if len(self.memory_pool):
            choices = np.random.choice(
                len(self.memory_pool),
                min(
                    len(self.memory_pool),
                    max(_num - len(self.tmp_memory_pool), 0)
                ),
                replace=False,
            )
        return [self.memory_pool[choice] for choice in choices] + self.tmp_memory_pool

    def merge(self):
        self.memory_pool += self.tmp_memory_pool
        self.tmp_memory_pool = []
        if len(self.memory_pool) > self.N:
            self.memory_pool = self.memory_pool[-self.N:]

    def __repr__(self) -> str:
        tmp = \
            "!!! tmp_memory_pool !!!\n" \
            "{}\n!!! memory_pool !!!\n" \
            "{}"
        return tmp.format(self.tmp_memory_pool, self.memory_pool)
