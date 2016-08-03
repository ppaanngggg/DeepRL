from multiprocessing import Process, Lock
import random
import sys
from select import select
import tensorflow as tf
from time import time
import zmq


def func_train_process(_create_agent_func,
                       _port, _lock,
                       _step_update_func):
    random.seed()
    agent = _create_agent_func()
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect('tcp://127.0.0.1:%s' % _port)

    def update_params():
        socket.send('params')
        fetch_data = socket.recv_pyobj()
        for k, v in zip(fetch_data.keys(), fetch_data.values()):
            if k == 'v_func':
                agent.setVFunc(v)
            elif k == 'q_func':
                agent.setQFunc(v)
            elif k == 'p_func':
                agent.setPFunc(v)
            elif k == 'target_v_func':
                agent.setTargetVFunc(v)
            elif k == 'target_q_func':
                agent.setTargetQFunc(v)
            elif k == 'target_p_func':
                agent.setTargetPFunc(v)
            else:
                raise Exception()

    def upload_grads_update_params():
        _lock.acquire()
        socket.send('grads')
        socket.recv()
        push_data = {}
        if agent.v_vars:
            push_data['v_func'] = agent.v_grads_data
        if agent.q_vars:
            push_data['q_func'] = agent.q_grads_data
        if agent.p_vars:
            push_data['p_func'] = agent.p_grads_data
        socket.send_pyobj(push_data)
        socket.recv()
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
            socket.send('step')
            tmp = socket.recv()
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
        context = zmq.Context()
        self.socket = context.socket(zmq.REP)
        port = self.socket.bind_to_random_port('tcp://127.0.0.1')

        self.lock = Lock()

        self.process_list = [
            Process(
                target=func_train_process,
                args=(_create_agent_func,
                      port, self.lock,
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
        start_time = time()
        while True:
            fetch_data = self.socket.recv()
            if fetch_data == 'step':
                self.socket.send('ack')
                # receive a step info, inc step_total
                self.step_total += 1
                if self.step_total % self.step_update_target == 0:
                    # if update target
                    self.agent.updateTargetFunc()
                    print time() - start_time
                    raw_input()
                if self.step_total % self.step_save == 0:
                    # if save model
                    self.agent.save("", self.step_total)
            elif fetch_data == 'params':
                # request params
                push_data = {}

                if self.agent.v_vars:
                    push_data['v_func'] = self.agent.getVFunc()
                if self.agent.q_vars:
                    push_data['q_func'] = self.agent.getQFunc()
                if self.agent.p_vars:
                    push_data['p_func'] = self.agent.getPFunc()
                if self.agent.target_v_vars:
                    push_data['target_v_func'] = self.agent.getTargetVFunc()
                if self.agent.target_q_vars:
                    push_data['target_q_func'] = self.agent.getTargetQFunc()
                if self.agent.target_p_vars:
                    push_data['target_p_func'] = self.agent.getTargetPFunc()
                self.socket.send_pyobj(push_data)
            elif fetch_data == 'grads':
                # get grads and update model
                self.socket.send('ack')
                fetch_data = self.socket.recv_pyobj()
                for k, v in zip(fetch_data.keys(), fetch_data.values()):
                    if k == 'v_func':
                        self.agent.v_grads_data = v
                    if k == 'q_func':
                        self.agent.q_grads_data = v
                    if k == 'p_func':
                        self.agent.p_grads_data = v
                self.agent.update()
                self.socket.send('ack')
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
