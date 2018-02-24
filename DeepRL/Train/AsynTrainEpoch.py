import logging
import sys
from select import select

import numpy as np
import torch.multiprocessing as mp
from DeepRL.Agent.AgentAbstract import AgentAbstract
from DeepRL.Env import EnvAbstract
from DeepRL.Train.TrainShell import TrainShell

logger = logging.getLogger()
logger.setLevel(logging.INFO)


class AsynTrainEpoch:
    """
    Asyn train for epoch algorithm like PPO

    _agent: the agent to be trained
    _env: the env object
    _epoch_max: the max epoch to play games
    _epoch_train: train model after how many games
    _train_update_target: update target after how many training
    _train_save: save model after how many training
    _process_core: how many cores to play games simultaneously
    _save_path: where to save models
    _use_cmd: whether to enable cmdtool while training
    """

    def __init__(
            self,
            _agent: AgentAbstract,
            _env: EnvAbstract,
            _epoch_max: int,
            _epoch_train: int,
            _train_update_target: int,
            _train_save: int,
            _process_core: int = None,
            _save_path: str = './save',
            _use_cmd: bool = True,
    ):
        self.agent: AgentAbstract = _agent
        self.agent.training()
        self.env: EnvAbstract = _env

        # multiprocessing for sampling
        self.mp = mp.get_context('spawn')
        self.process_core = _process_core
        self.pool = self.mp.Pool(self.process_core)

        # training control
        self.epoch = 0
        self.train_times = 0
        self.epoch_max = _epoch_max
        self.epoch_train = _epoch_train
        self.train_update_target = _train_update_target
        self.train_save = _train_save

        self.total_reward_buf = []

        self.save_path = _save_path
        self.use_cmd = _use_cmd
        if self.use_cmd:
            self.shell = TrainShell(self)

    @staticmethod
    def loop_env(
            _agent: AgentAbstract, _env: EnvAbstract, _epoch_num: int
    ):
        logger.info('Start new game: {}'.format(_epoch_num))

        _agent.startNewGame()
        while _agent.step():
            pass

        return (
            _agent.getDataset(_agent.replay.pull()),
            _env.total_reward
        )

    @staticmethod
    def merge_dataset_reward(_ret_list):
        dataset_list = [d[0] for d in _ret_list]
        tuple_len = len(dataset_list[0])
        dataset = []
        for i in range(tuple_len):
            dataset.append(np.concatenate([
                tmp[i] for tmp in dataset_list
            ]))
        return dataset, [d[1] for d in _ret_list]

    def run(self):
        while self.epoch < self.epoch_max:
            # multiprocessing to get dataset
            ret_list = self.pool.starmap(
                AsynTrainEpoch.loop_env,
                [(self.agent, self.env, tmp) for tmp in range(
                    self.epoch, self.epoch + self.epoch_train
                )]
            )
            self.epoch += self.epoch_train

            # train model
            dataset, rewards = AsynTrainEpoch.merge_dataset_reward(ret_list)
            self.agent.train(dataset)
            self.total_reward_buf.append(rewards)

            self.train_times += 1
            if not self.train_times % self.train_update_target:
                self.agent.updateTargetFunc()
            if not self.train_times % self.train_save:
                self.agent.save(
                    self.epoch, 0, self.save_path
                )

            if self.use_cmd:
                rlist, _, _ = select([sys.stdin], [], [], 0.0)
                if rlist:
                    sys.stdin.readline()
                    self.shell.cmdloop()
                else:
                    pass
