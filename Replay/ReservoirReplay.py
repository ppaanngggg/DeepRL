import numpy as np
from Replay import ReplayTuple


class ReservoirReplay():

    def __init__(self, _N=1e4):
        self.N = int(_N)
        self.memory_pool = []

    def push(self, _state, _action, _reward, _next_state, _mask=None):
        if len(self.memory_pool) == self.N:
            idx = np.random.randint(0, self.N)
            del self.memory_pool[idx]
        # store new tuples into tmp memory buffer
        self.tmp_memory_pool.append(
            ReplayTuple(_state, _action, _reward, _next_state, _mask)
        )

    def pull(self, _num):
        choices = []
        if len(self.memory_pool):
            choices = np.random.choice(
                len(self.memory_pool),
                min(len(self.memory_pool), _num),
                replace=False,
            )
        return [self.memory_pool[choice] for choice in choices], None

    def setErr(self, _batch_tuples, _err_list):
        pass

    def merge(self):
        pass

    def show(self):
        print '!!! tmp_memory_pool !!!'
        for t in self.tmp_memory_pool:
            t.show()
        print '!!! memory_pool !!!'
        for t in self.memory_pool:
            t.show()
