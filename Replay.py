import numpy as np
import Config

class ReplayTuple:

    def __init__(self, _state, _action, _reward, _next_state, _mask):
        self.state = _state
        self.action = _action
        self.reward = _reward
        self.next_state = _next_state
        self.mask = _mask
        self.P = None
        self.p = None
        self.err = None

    def show(self):
        print '----- begin -----'
        self.state.show()
        self.next_state.show()
        print 'mask:', self.mask
        print 'action:', self.action
        print 'reward:', self.reward
        print 'P:', self.P
        print 'p:', self.p
        print 'err:', self.err
        print '----- end -----'


class Replay():

    def __init__(self):
        self.N = Config.replay_N
        self.memory_pool = []
        self.tmp_memory_pool = []

    def push(self, _tuple):
        # store new tuples into tmp memory buffer
        if len(self.memory_pool):
            _tuple.P = max([t.P for t in self.memory_pool])
        else:
            _tuple.P = 1.
        self.tmp_memory_pool.append(_tuple)

    def pull(self, _num):
        # if memory_pool is not empty choose from pool according to Pi
        choices = []
        if len(self.memory_pool):
            choices = np.random.choice(
                len(self.memory_pool),
                min(len(self.memory_pool), max(
                    _num - len(self.tmp_memory_pool), 0)),
                replace=False,
                p=[t.P for t in self.memory_pool]
            )
        return [self.memory_pool[choice] for choice in choices] + self.tmp_memory_pool

    def merge(self, _alpha):
        self.memory_pool += self.tmp_memory_pool
        self.tmp_memory_pool = []
        # sort memory pool by err
        self.memory_pool = sorted(
            self.memory_pool, key=lambda x: x.err, reverse=True)
        if len(self.memory_pool) > self.N:
            self.memory_pool = self.memory_pool[:self.N]
        # compute pi = 1 / rank(i), and count sum(pi^alpha)
        total_p = 0.
        for i in range(len(self.memory_pool)):
            self.memory_pool[i].p = 1. / (i + 1.)
            total_p += self.memory_pool[i].p ** _alpha
        # compute Pi
        for i in range(len(self.memory_pool)):
            self.memory_pool[i].P = self.memory_pool[i].p ** _alpha / total_p

    def show(self):
        print '!!! tmp_memory_pool !!!'
        for t in self.tmp_memory_pool:
            t.show()
        print '!!! memory_pool !!!'
        for t in self.memory_pool:
            t.show()

    def getPoolSize(self):
        return len(self.memory_pool)
