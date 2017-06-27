from ..Model.ACModel import Actor, Critic
from Agent import Agent
import random
from chainer import serializers, Variable
import chainer.functions as F
import numpy as np

import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


class NFSPAgent(Agent):

    def __init__(self, _shared, _actor, _critic, _env, _is_train=True,
                 _actor_optimizer=None, _critic_optimizer=None,
                 _actor_replay=None, _critic_replay=None,
                 _gpu=False, _gamma=0.99, _batch_size=32,
                 _epsilon=0.5, _epsilon_decay=0.995, _epsilon_underline=0.01,
                 _grad_clip=1., _eta=0.1):
        """
        Args:
            _shared (class):
            _actor (class):
            _critic (class):
        """
        super(NFSPAgent, self).__init__()

        self.is_train = _is_train

        self.p_func = Actor(_shared(), _actor())
        self.q_func = Critic(_shared(), _critic())
        self.env = _env

        if self.is_train:
            self.target_q_func = Critic(_shared(), _critic())
            self.target_q_func.copyparams(self.q_func)

            if _actor_optimizer:
                self.p_opt = _actor_optimizer
                self.p_opt.setup(self.p_func)
            if _critic_optimizer:
                self.v_opt = _critic_optimizer
                self.v_opt.setup(self.q_func)

            self.p_replay = _actor_replay
            self.q_replay = _critic_replay
            self.replay = self.q_replay

        self.config.eta = _eta
        self.config.gpu = _gpu
        self.config.gamma = _gamma
        self.config.batch_size = _batch_size
        self.config.epsilon = _epsilon
        self.config.epsilon_decay = _epsilon_decay
        self.config.epsilon_underline = _epsilon_underline
        self.config.grad_clip = _grad_clip

    def startNewGame(self):
        if self.is_train:
            if random.random() < self.config.eta:
                self.use_func = self.q_func
            else:
                self.use_func = self.p_func
        else:
            self.use_func = self.p_func
        super(NFSPAgent, self).startNewGame()

    def step(self):
        """
        Returns:
            still in game or not
        """
        if not self.env.in_game:
            return False

        # get current state
        cur_state = self.env.getState()
        # choose action in step
        action = self.chooseAction(self.use_func, cur_state)
        # do action and get reward
        reward = self.env.doAction(action)

        logger.info('Action: ' + str(action) + '; Reward: %.3f' % (reward))

        if self.is_train:
            # get new state
            next_state = self.env.getState()
            # store replay_tuple into memory pool
            self.q_replay.push(cur_state, action, reward, next_state)
            if self.use_func == self.q_func:
                self.p_replay.push(cur_state, action, None, None)

        return self.env.in_game

    def chooseAction(self, _model, _state):
        if self.is_train:
            # update epsilon
            if _model == self.q_func:
                self.updateEpsilon()
                if random.random() < self.config.epsilon:
                    # randomly choose
                    return self.env.getRandomAction(_state)
            # use model to choose
            x_data = self.env.getX(_state)
            output = self.func(_model, x_data, False)
            return self.env.getBestAction(output.data, [_state])[0]
        else:
            x_data = self.env.getX(_state)
            output = self.func(_model, x_data, False)
            logger.info(str(output.data))
            return self.env.getBestAction(output.data, [_state])[0]

    def forward(self, _cur_x, _next_x, _state_list):
        # get cur outputs
        cur_output = self.func(self.q_func, _cur_x, True)
        # get next outputs, NOT target
        next_output = self.func(self.q_func, _next_x, False)

        # only one head
        next_action = self.env.getBestAction(
            next_output.data, _state_list)

        # get next outputs, target
        next_output = self.func(self.target_q_func, _next_x, False)
        return cur_output, next_output, next_action

    def grad(self, _cur_output, _next_output, _next_action, _batch_tuples):
        # alloc
        if self.config.gpu:
            _cur_output.grad = cupy.zeros_like(_cur_output.data)
        else:
            _cur_output.grad = np.zeros_like(_cur_output.data)

        # compute grad from each tuples
        err_list = []
        for i in range(len(_batch_tuples)):
            cur_action_value = \
                _cur_output.data[i][_batch_tuples[i].action].tolist()
            reward = _batch_tuples[i].reward
            target_value = reward
            # if not empty position, not terminal state
            if _batch_tuples[i].next_state.in_game:
                next_action_value = \
                    _next_output.data[i][_next_action[i]].tolist()
                target_value += self.config.gamma * next_action_value
            loss = cur_action_value - target_value
            _cur_output.grad[i][_batch_tuples[i].action] = 2 * loss
            err_list.append(abs(loss))
        return err_list

    def doTrain(self, _batch_tuples, _weights):
        # get inputs from batch
        cur_x = self.getCurInputs(_batch_tuples)
        next_x = self.getNextInputs(_batch_tuples)
        # compute forward
        cur_output, next_output, next_action = self.forward(
            cur_x, next_x, [t.next_state for t in _batch_tuples])
        # fill grad
        err_list = self.grad(cur_output, next_output,
                             next_action, _batch_tuples)
        if _weights is not None:
            self.gradWeight(cur_output, _weights)
        if self.config.grad_clip:
            self.gradClip(cur_output, self.config.grad_clip)
        # backward
        cur_output.backward()

        return err_list

    def train(self):
        super(NFSPAgent, self).train()

        batch_tuples, _ = self.p_replay.pull(self.config.batch_size)
        if len(batch_tuples) > 0:
            x = self.getCurInputs(batch_tuples)
            t = np.array([d.action for d in batch_tuples], np.int32)
            y = self.p_func(x)
            loss = F.softmax_cross_entropy(y, t)
            loss.backward()
            self.p_opt.update()
