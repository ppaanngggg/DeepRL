from Replay import ReplayTuple


class TmpReplay(object):

    def __init__(self):
        self.tmp_memory_pool = []

    def push(self, _state, _action, _reward, _next_state, _mask=None):
        # store new tuples into tmp memory buffer
        self.tmp_memory_pool.append(
            ReplayTuple(_state, _action, _reward, _next_state, _mask)
        )

    def pull(self, _num):
        return self.tmp_memory_pool, None

    def setErr(self, _batch_tuples, _err_list):
        pass

    def merge(self):
        self.tmp_memory_pool = []

    def show(self):
        print '!!! tmp_memory_pool !!!'
        for t in self.tmp_memory_pool:
            t.show()
