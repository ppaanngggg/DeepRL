import numpy as np


class PrioritizedReplayTuple:

    def __init__(self, _state, _action, _reward, _next_state, _mask=None):
        self.state = _state
        self.action = _action
        self.reward = _reward
        self.next_state = _next_state
        self.P = None
        self.p = None
        self.err = None
        self.mask = _mask

    def show(self):
        print '----- begin -----'
        self.state.show()
        self.next_state.show()
        print 'action:', self.action
        print 'reward:', self.reward
        print 'P:', self.P
        print 'p:', self.p
        print 'err:', self.err
        if self.mask is not None:
            print 'mask:', self.mask
        print '----- end -----'


class PrioritizedReplay():

    def __init__(self, _N=1e4, _alpha=0.7, _beta=0.5, _beta_add=1e-4):
        self.N = int(_N)
        self.alpha = _alpha
        self.beta = _beta
        self.beta_add = _beta_add

        self.memory_pool = []
        self.tmp_memory_pool = []

    def push(self, _state, _action, _reward, _next_state, _mask):
        _tuple = PrioritizedReplayTuple(
            _state, _action, _reward, _next_state, _mask)
        # if use prioritized_replay, need to init P for new tuples
        if len(self.memory_pool):
            _tuple.P = max([t.P for t in self.memory_pool])
        else:
            _tuple.P = 1.
        # store new tuples into tmp memory buffer
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
        batch_tuples = [self.memory_pool[choice]
                        for choice in choices] + self.tmp_memory_pool
        return batch_tuples, self.getWeights(batch_tuples)

    def getWeights(self, _batch_tuples):
        if not len(_batch_tuples):
            return None
        # compute grad's weights
        weights = np.array([t.P for t in _batch_tuples], np.float32)
        if self.getPoolSize():
            weights *= self.getPoolSize()
        weights = weights ** -self.beta
        weights /= weights.max()
        weights = np.expand_dims(weights, 1)
        # update beta
        self.beta = min(1, self.beta + self.beta_add)

        return weights

    def setErr(self, _batch_tuples, _err_list):
        for t, e in zip(_batch_tuples, _err_list):
            if e is not None:
                t.err = e

    def merge(self):
        self.memory_pool += self.tmp_memory_pool
        self.tmp_memory_pool = []

        # sort memory pool by err
        self.memory_pool = sorted(
            self.memory_pool, key=lambda x: x.err, reverse=True)
        # pop last
        if len(self.memory_pool) > self.N:
            self.memory_pool = self.memory_pool[:self.N]
        # compute pi = 1 / rank(i), and count sum(pi^alpha)
        total_p = 0.
        for i in range(len(self.memory_pool)):
            self.memory_pool[i].p = 1. / (i + 1.)
            total_p += self.memory_pool[i].p ** self.alpha
        # compute Pi
        for i in range(len(self.memory_pool)):
            self.memory_pool[i].P = \
                self.memory_pool[i].p ** self.alpha / total_p

    def show(self):
        print '!!! tmp_memory_pool !!!'
        for t in self.tmp_memory_pool:
            t.show()
        print '!!! memory_pool !!!'
        for t in self.memory_pool:
            t.show()

    def getPoolSize(self):
        return len(self.memory_pool)
