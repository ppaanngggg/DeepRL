from multiprocessing import Process, Queue, Lock
import random
import sys
from select import select
import tensorflow as tf
from time import time


def func_train_process(_create_agent_func,
                       _c2s_queue, _s2c_queue, _lock,
                       _step_update_func):
    random.seed()
    agent = _create_agent_func()

    def set_params(_vars, _params):
        with tf.device(agent.config.device):
            agent.sess.run([
                v.assign(p) for v, p in zip(_vars, _params)
            ])

    def update_params():
        _c2s_queue.put('params')
        fetch_data = _s2c_queue.get()
        for k, v in zip(fetch_data.keys(), fetch_data.values()):
            if k == 'v_func':
                set_params(agent.v_vars, v)
            elif k == 'q_func':
                set_params(agent.q_vars, v)
            elif k == 'p_func':
                set_params(agent.p_vars, v)
            elif k == 'target_v_func':
                set_params(agent.target_v_vars, v)
            elif k == 'target_q_func':
                start_time = time()
                set_params(agent.target_q_vars, v)
                print 'update_params time', time() - start_time
            elif k == 'target_p_func':
                set_params(agent.target_p_vars, v)
            else:
                raise Exception()

    def upload_grads_update_params():
        _lock.acquire()
        _c2s_queue.put('grads')
        _s2c_queue.get()
        push_data = {}
        if agent.v_vars:
            push_data['v_func'] = agent.v_grads_data
        if agent.q_vars:
            push_data['q_func'] = agent.q_grads_data
        if agent.p_vars:
            push_data['p_func'] = agent.p_grads_data
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

        self.step_total = 0
        self.step_update_target = _step_update_target
        self.step_save = _step_save

    def run(self):
        while True:
            fetch_data = self.c2s_queue.get()
            if fetch_data == 'step':
                # receive a step info, inc step_total
                self.step_total += 1
                if self.step_total % self.step_update_target == 0:
                    # if update target
                    self.agent.updateTargetFunc()
                if self.step_total % self.step_save == 0:
                    # if save model
                    self.agent.save("", self.step_total)
                self.s2c_queue.put('ack')
            elif fetch_data == 'params':
                # request params
                push_data = {}

                def get_vars_params(_vars):
                    return self.agent.sess.run(_vars)

                if self.agent.v_vars:
                    push_data['v_func'] = get_vars_params(self.agent.v_vars)
                if self.agent.q_vars:
                    push_data['q_func'] = get_vars_params(self.agent.q_vars)
                if self.agent.p_vars:
                    push_data['p_func'] = get_vars_params(self.agent.p_vars)
                if self.agent.target_v_vars:
                    push_data['target_v_func'] = get_vars_params(
                        self.agent.target_v_vars)
                if self.agent.target_q_vars:
                    push_data['target_q_func'] = get_vars_params(
                        self.agent.target_q_vars)
                if self.agent.target_p_vars:
                    push_data['target_p_func'] = get_vars_params(
                        self.agent.target_p_vars)
                self.s2c_queue.put(push_data)
            elif fetch_data == 'grads':
                # get grads and update model
                self.s2c_queue.put('ack')
                fetch_data = self.c2s_queue.get()
                for k, v in zip(fetch_data.keys(), fetch_data.values()):
                    if k == 'v_func':
                        self.agent.v_grads_data = v
                    if k == 'q_func':
                        self.agent.q_grads_data = v
                    if k == 'p_func':
                        self.agent.p_grads_data = v
                self.agent.update()
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
