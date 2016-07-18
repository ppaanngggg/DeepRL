from multiprocessing import Process, Queue, Lock
import random
import sys
from select import select
from chainer import cuda


def func_train_process(_create_agent_func,
                       _c2s_queue, _s2c_queue, _lock,
                       _step_update_func):
    random.seed()
    agent = _create_agent_func()

    def set_params(_func, _params):
        for param, data in zip(_func.params(), _params):
            param.data = data

    def update_params():
        _c2s_queue.put('params')
        fetch_data = _s2c_queue.get()
        for k, v in zip(fetch_data.keys(), fetch_data.values()):
            if k == 'v_func':
                set_params(agent.v_func, v)
            if k == 'q_func':
                set_params(agent.q_func, v)
            if k == 'p_func':
                set_params(agent.p_func, v)
            if k == 'target_v_func':
                set_params(agent.target_v_func, v)
            if k == 'target_q_func':
                set_params(agent.target_q_func, v)
            if k == 'target_p_func':
                set_params(agent.target_p_func, v)

    def upload_grads_update_params():
        _lock.acquire()
        _c2s_queue.put('grads')
        _s2c_queue.get()
        push_data = {}
        if agent.v_func:
            push_data['v_func'] = [d.grad for d in agent.v_func.params()]
        if agent.q_func:
            push_data['q_func'] = [d.grad for d in agent.q_func.params()]
        if agent.p_func:
            push_data['p_func'] = [d.grad for d in agent.p_func.params()]
        _c2s_queue.put(push_data)
        _s2c_queue.get()
        update_params()
        _lock.release()

    _lock.acquire()
    update_params()
    _lock.release()

    while True:
        agent.startNewGame()
        step_local = 0
        while agent.step():
            _lock.acquire()
            _c2s_queue.put('step')
            _s2c_queue.get()
            _lock.release()
            step_local += 1
            if step_local % _step_update_func == 0:
                agent.train()
                upload_grads_update_params()
        agent.train()
        upload_grads_update_params()


class AsynTrain(object):

    def __init__(self, _create_agent_func, _process_num=8,
                 _step_update_func=5,
                 _step_update_target=1e3,
                 _step_save=1e6,
                 _v_opt=None, _q_opt=None, _p_opt=None):
        self.c2s_queue = Queue()
        self.s2c_queue = Queue()
        self.lock = Lock()

        self.process_list = [
            Process(
                target=func_train_process,
                args=(_create_agent_func,
                      self.c2s_queue, self.s2c_queue, self.lock,
                      _step_update_func))
            for _ in range(_process_num)
        ]
        for process in self.process_list:
            process.start()

        self.agent = _create_agent_func()
        self.v_func = self.agent.v_func
        self.q_func = self.agent.q_func
        self.p_func = self.agent.p_func
        self.target_v_func = self.agent.target_v_func
        self.target_q_func = self.agent.target_q_func
        self.target_p_func = self.agent.target_p_func

        self.v_opt = _v_opt
        self.q_opt = _q_opt
        self.p_opt = _p_opt
        if _v_opt and self.v_func:
            self.v_opt.setup(self.v_func)
        if _q_opt and self.q_func:
            self.q_opt.setup(self.q_func)
        if _p_opt and self.p_func:
            self.p_opt.setup(self.p_func)

        self.step_total = 0
        self.step_update_target = _step_update_target
        self.step_save = _step_save

    def run(self):
        while True:
            fetch_data = self.c2s_queue.get()
            if fetch_data == 'step':
                self.step_total += 1
                if self.step_total % self.step_update_target == 0:
                    self.agent.updateTargetFunc()
                if self.step_total % self.step_save == 0:
                    self.agent.save("", self.step_total)
                self.s2c_queue.put('ack')
            elif fetch_data == 'params':
                push_data = {}
                if self.v_func:
                    push_data['v_func'] = [
                        d.data for d in self.v_func.params()]
                if self.q_func:
                    push_data['q_func'] = [
                        d.data for d in self.q_func.params()]
                if self.p_func:
                    push_data['p_func'] = [
                        d.data for d in self.p_func.params()]
                if self.target_v_func:
                    push_data['target_v_func'] = [
                        d.data for d in self.target_v_func.params()]
                if self.target_q_func:
                    push_data['target_q_func'] = [
                        d.data for d in self.target_q_func.params()]
                if self.target_p_func:
                    push_data['target_p_func'] = [
                        d.data for d in self.target_p_func.params()]
                self.s2c_queue.put(push_data)
            elif fetch_data == 'grads':
                self.s2c_queue.put('ack')
                fetch_data = self.c2s_queue.get()
                for k, v in zip(fetch_data.keys(), fetch_data.values()):
                    if k == 'v_func':
                        for param, grad in zip(self.v_func.params(), v):
                            param.grad = grad
                    if k == 'q_func':
                        for param, grad in zip(self.q_func.params(), v):
                            param.grad = grad
                    if k == 'p_func':
                        for param, grad in zip(self.p_func.params(), v):
                            param.grad = grad
                if self.v_opt:
                    self.v_opt.update()
                if self.q_opt:
                    self.q_opt.update()
                if self.p_opt:
                    self.p_opt.update()
                self.s2c_queue.put('ack')
            else:
                raise Exception()

            # cmd
            rlist, _, _ = select([sys.stdin], [], [], 0.001)
            if rlist:
                print '[[[ interrupted ]]]'
                s = sys.stdin.readline().strip()
                while True:
                    print '[[[ Please input (save, quit, ...) ]]]'
                    s = sys.stdin.readline().strip()
                    if s == 'save':
                        self.agent.save("", self.step_total)
                    elif s == 'quit':
                        break
                    else:
                        print '[[[ unknow cmd... ]]]'
                        pass
            else:
                pass
