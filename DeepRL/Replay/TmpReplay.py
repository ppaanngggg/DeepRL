import typing

from DeepRL.Env import EnvState
from DeepRL.Replay.ReplayAbstract import ReplayAbstract, ReplayTuple


class TmpReplay(ReplayAbstract):
    def __init__(self):
        self.memory_pool: typing.List[ReplayTuple] = []

    def push(
            self, _state: EnvState,
            _action: int, _reward: float,
            _next_state: EnvState
    ):
        # store new tuples into tmp memory buffer
        self.memory_pool.append(
            ReplayTuple(_state, _action, _reward, _next_state)
        )

    def pull(
            self, _num: int = None
    ) -> typing.Sequence[ReplayTuple]:
        return self.memory_pool

    def merge(self):
        self.memory_pool = []

    def __repr__(self) -> str:
        return '{}'.format(self.memory_pool)
