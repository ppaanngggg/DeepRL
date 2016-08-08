from multiprocessing import Process, Lock
import random
import sys
from select import select
import tensorflow as tf
import numpy as np
import zmq
import SharedArray as sa
from time import time


def func_train_process(_create_agent_func, _port,
                       _vars_lock, _grads_lock, _step_update_func):
    random.seed()
    agent = _create_agent_func()
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect('tcp://127.0.0.1:%s' % _port)

    socket.send('shared')
    shared_vars_name_dict, shared_grads_name_dict = socket.recv_pyobj()
    shared_vars_dict = {}
    for k, v in zip(shared_vars_name_dict.keys(), shared_vars_name_dict.values()):
        shared_vars_dict[k] = []
        for _v in v:
            shared_vars_dict[k].append(sa.attach(_v))
    shared_grads_dict = {}
    for k, v in zip(shared_grads_name_dict.keys(), shared_grads_name_dict.values()):
        shared_grads_dict[k] = []
        for _v in v:
            shared_grads_dict[k].append(sa.attach(_v))

    def update_params():
        _vars_lock.acquire()
        for k, v in zip(shared_vars_dict.keys(), shared_vars_dict.values()):
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
        _vars_lock.release()

    def setGrads(_key, _grads):
        for d, g in zip(shared_grads_dict[_key], _grads):
            np.copyto(d, g)

    def upload_grads():
        _grads_lock.acquire()
        if agent.v_vars:
            setGrads('v_func', agent.v_grads_data)
        if agent.q_vars:
            setGrads('q_func', agent.q_grads_data)
        if agent.p_vars:
            setGrads('p_func', agent.p_grads_data)
        socket.send('grads')
        socket.recv()
        _grads_lock.release()

    update_params()

    while True:
        agent.startNewGame()
        step_local = 0
        while True:
            in_game = agent.step()
            step_local += 1
            if not in_game or step_local % _step_update_func == 0:
                agent.train()
                upload_grads()
            socket.send('step')
            socket.recv()
            if not in_game or step_local % _step_update_func == 0:
                update_params()
            if not in_game:
                break


class AsynTrain(object):

    def __init__(self, _create_agent_func, _process_num=8,
                 _step_update_func=5,
                 _step_update_target=1e3,
                 _step_save=1e6,
                 _v_opt=None, _q_opt=None, _p_opt=None):
        # create socket to connect actors
        context = zmq.Context()
        self.socket = context.socket(zmq.REP)
        port = self.socket.bind_to_random_port('tcp://127.0.0.1')
        # lock when vars is being written
        self.vars_lock = Lock()
        # lock when upload grads
        self.grads_lock = Lock()

        # start actors
        self.process_list = [
            Process(
                target=func_train_process,
                args=(_create_agent_func, port,
                      self.vars_lock, self.grads_lock, _step_update_func))
            for _ in range(_process_num)
        ]
        for process in self.process_list:
            process.start()

        # create agent, and create optimizer
        self.agent = _create_agent_func()
        if _v_opt is not None:
            self.agent.createVOpt(_v_opt)
        if _q_opt is not None:
            self.agent.createQOpt(_q_opt)
        if _p_opt is not None:
            self.agent.createPOpt(_p_opt)

        self.agent.sess.run(tf.initialize_all_variables())
        self.agent.updateTargetFunc()

        # alloc mem for vars
        self.shared_vars_dict = {}
        self.shared_vars_name_dict = {}

        def createSharedVars(_name, _data_list):
            self.shared_vars_dict[_name] = []
            self.shared_vars_name_dict[_name] = []
            for i in range(len(_data_list)):
                array_name = 'shm://' + _name + '_' + str(i)
                self.shared_vars_name_dict[_name].append(array_name)
                array = sa.create(array_name, _data_list[i].shape, np.float32)
                np.copyto(array, _data_list[i])
                self.shared_vars_dict[_name].append(array)

        # alloc mem for grads
        self.shared_grads_dict = {}
        self.shared_grads_name_dict = {}

        def createSharedGrads(_name, _data_list):
            self.shared_grads_dict[_name] = []
            self.shared_grads_name_dict[_name] = []
            for i in range(len(_data_list)):
                array_name = 'shm://' + _name + '_grad_' + str(i)
                self.shared_grads_name_dict[_name].append(array_name)
                array = sa.create(array_name, _data_list[i].shape, np.float32)
                self.shared_grads_dict[_name].append(array)

        if self.agent.v_vars:
            createSharedVars('v_func', self.agent.getVFunc())
            createSharedGrads('v_func', self.agent.getVFunc())
        if self.agent.q_vars:
            createSharedVars('q_func', self.agent.getQFunc())
            createSharedGrads('q_func', self.agent.getQFunc())
        if self.agent.p_vars:
            createSharedVars('p_func', self.agent.getPFunc())
            createSharedGrads('p_func', self.agent.getPFunc())
        if self.agent.target_v_vars:
            createSharedVars('target_v_func', self.agent.getTargetVFunc())
        if self.agent.target_q_vars:
            createSharedVars('target_q_func', self.agent.getTargetQFunc())
        if self.agent.target_p_vars:
            createSharedVars('target_p_func', self.agent.getTargetPFunc())

        self.step_total = 0
        self.step_update_target = _step_update_target
        self.step_save = _step_save

    def run(self):
        def setVars(_key, _vars):
            for d, v in zip(self.shared_vars_dict[_key], _vars):
                np.copyto(d, v)

        while True:
            cmd = self.socket.recv()
            if cmd == 'shared':
                self.socket.send_pyobj(
                    (self.shared_vars_name_dict, self.shared_grads_name_dict)
                )
            elif cmd == 'step':
                # send ack
                self.step_total += 1
                if self.step_total % self.step_update_target == 0:
                    # if update target
                    self.vars_lock.acquire()
                    self.agent.updateTargetFunc()
                    if self.agent.target_v_vars:
                        setVars('target_v_func', self.agent.getTargetVFunc())
                    if self.agent.target_q_vars:
                        setVars('target_q_func', self.agent.getTargetQFunc())
                    if self.agent.target_p_vars:
                        setVars('target_p_func', self.agent.getTargetPFunc())
                    self.vars_lock.release()
                self.socket.send('ack')
                if self.step_total % self.step_save == 0:
                    # if save model
                    self.agent.save("", self.step_total)
            elif cmd == 'grads':
                # get grads and update model
                for k, g in zip(self.shared_grads_dict.keys(), self.shared_grads_dict.values()):
                    if k == 'v_func':
                        self.agent.v_grads_data = g
                    if k == 'q_func':
                        self.agent.q_grads_data = g
                    if k == 'p_func':
                        self.agent.p_grads_data = g
                self.agent.update()

                self.vars_lock.acquire()
                if self.agent.v_vars:
                    setVars('v_func', self.agent.getVFunc())
                if self.agent.q_vars:
                    setVars('q_func', self.agent.getQFunc())
                if self.agent.p_vars:
                    setVars('p_func', self.agent.getPFunc())
                self.vars_lock.release()

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
