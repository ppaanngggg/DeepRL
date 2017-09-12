import logging
import sys
from select import select

import numpy as np
import torch.multiprocessing as mp

from DeepRL.Agent.AgentAbstract import AgentAbstract
from DeepRL.Train.TrainShell import TrainShell

logger = logging.getLogger()
logger.setLevel(logging.INFO)


class AsynTrainEpoch:
    def __init__(
            self,
            _agent: AgentAbstract,
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

        self.mp = mp.get_context('spawn')
        self.process_core = _process_core
        self.pool = self.mp.Pool(self.process_core)

        self.epoch = 0
        self.epoch_max = _epoch_max
        self.epoch_train = _epoch_train
        self.train_update_target = _train_update_target
        self.train_save = _train_save

        self.save_path = _save_path
        self.use_cmd = _use_cmd
        if self.use_cmd:
            self.shell = TrainShell(self)

    @staticmethod
    def loop_env(_agent: AgentAbstract, _epoch_num: int):
        logger.info('Start new game: {}'.format(_epoch_num))

        _agent.startNewGame()
        while _agent.step():
            pass

        return _agent.getDataset(_agent.replay.pull())

    @staticmethod
    def merge_dataset(_dataset_list):
        tuple_len = len(_dataset_list[0])
        dataset = []
        for i in range(tuple_len):
            dataset.append(np.concatenate([
                tmp[i] for tmp in _dataset_list
            ]))
        return dataset

    def run(self):
        self.train_times = 0
        while self.epoch < self.epoch_max:
            # multiprocessing to get dataset
            dataset_list = self.pool.starmap(
                AsynTrainEpoch.loop_env,
                [(self.agent, tmp) for tmp in range(
                    self.epoch, self.epoch + self.epoch_train
                )]
            )
            self.epoch += self.epoch_train

            # train model
            dataset = AsynTrainEpoch.merge_dataset(dataset_list)
            self.agent.train(dataset)

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
