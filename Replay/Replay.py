import numpy as np


class ReplayTuple:

    def __init__(self, _state, _action, _reward, _next_state):
        self.state = _state
        self.action = _action
        self.reward = _reward
        self.next_state = _next_state

    def show(self):
        print '----- begin -----'
        self.state.show()
        self.next_state.show()
        print 'action:', self.action
        print 'reward:', self.reward
        print '----- end -----'


class Replay():

    def __init__(self, _N=1e4):
        self.N = int(_N)
        self.memory_pool = []
        self.tmp_memory_pool = []

    def push(self, _state, _action, _reward, _next_state):
        # store new tuples into tmp memory buffer
        self.tmp_memory_pool.append(
            ReplayTuple(_state, _action, _reward, _next_state)
        )

    def pull(self, _num):
        choices = []
        if len(self.memory_pool):
            choices = np.random.choice(
                len(self.memory_pool),
                min(len(self.memory_pool),
                    max(_num - len(self.tmp_memory_pool), 0)),
                replace=False,
            )
        return [self.memory_pool[choice] for choice in choices] + self.tmp_memory_pool

    def merge(self):
        self.memory_pool += self.tmp_memory_pool
        self.tmp_memory_pool = []
        if len(self.memory_pool) > self.N:
            self.memory_pool = self.memory_pool[-self.N:]

    def show(self):
        print '!!! tmp_memory_pool !!!'
        for t in self.tmp_memory_pool:
            t.show()
        print '!!! memory_pool !!!'
        for t in self.memory_pool:
            t.show()
