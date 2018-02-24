import logging
import sys
from select import select

from DeepRL.Agent.AgentAbstract import AgentAbstract
from DeepRL.Env import EnvAbstract
from DeepRL.Train.TrainShell import TrainShell

logger = logging.getLogger()
logger.setLevel(logging.INFO)


class TrainEpoch:
    def __init__(
            self,
            _agent: AgentAbstract,
            _env: EnvAbstract,
            _epoch_max: int,
            _epoch_train: int,
            _train_update_target: int,
            _train_save: int,
            _save_path: str = './save',
            _use_cmd: bool = True,
    ):
        self.agent: AgentAbstract = _agent
        self.agent.training()  # set to training mode

        self.env = _env

        self.epoch = 0
        self.train_time = 0
        self.epoch_max = _epoch_max
        self.epoch_train = _epoch_train
        self.train_update_target = _train_update_target
        self.train_save = _train_save

        self.total_reward_buf = []

        self.save_path = _save_path
        self.use_cmd = _use_cmd
        if self.use_cmd:
            self.shell = TrainShell(self)

    def run(self):
        tmp_reward_buf = []
        while self.epoch < self.epoch_max:
            logger.info('Start new game: {}'.format(self.epoch))
            # collect data
            for _ in range(self.epoch_train):
                self.agent.startNewGame()
                self.epoch += 1

                # step until game finishes
                while self.agent.step():
                    pass
                tmp_reward_buf.append(self.env.total_reward)

                if self.use_cmd:  # cmd
                    rlist, _, _ = select([sys.stdin], [], [], 0.0)
                    if rlist:
                        sys.stdin.readline()
                        self.shell.cmdloop()
                    else:
                        pass

            # train model
            self.agent.train()
            self.train_time += 1
            self.total_reward_buf.append(tmp_reward_buf)
            tmp_reward_buf = []

            if not self.train_time % self.train_update_target:
                self.agent.updateTargetFunc()
            if not self.train_time % self.train_save:
                self.agent.save(
                    self.epoch, 0, self.save_path
                )
