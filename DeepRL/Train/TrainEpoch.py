import logging
import sys
from select import select

from DeepRL.Agent.AgentAbstract import AgentAbstract
from DeepRL.Train.TrainShell import TrainShell

logger = logging.getLogger()
logger.setLevel(logging.INFO)


class TrainEpoch:
    def __init__(
            self,
            _agent: AgentAbstract,
            _epoch_max: int,
            _epoch_train: int,
            _epoch_update_target: int,
            _epoch_save: int,
            _save_path: str = './save',
            _use_cmd: bool = True,
    ):
        self.agent: AgentAbstract = _agent
        self.agent.training()  # set to training mode

        self.epoch = 0
        self.epoch_max = _epoch_max
        self.epoch_train = _epoch_train
        self.epoch_update_target = _epoch_update_target
        self.epoch_save = _epoch_save

        self.save_path = _save_path
        self.use_cmd = _use_cmd
        if self.use_cmd:
            self.shell = TrainShell(self)

    def run(self):
        while self.epoch < self.epoch_max:
            logger.info('Start new game: {}'.format(self.epoch))

            self.agent.startNewGame()
            self.epoch += 1

            # step until game finishes
            while self.agent.step():
                pass

            if not self.epoch % self.epoch_train:
                self.agent.train()
            if not self.epoch % self.epoch_update_target:
                self.agent.updateTargetFunc()
            if not self.epoch % self.epoch_save:
                self.agent.save(
                    self.epoch, 0, self.save_path
                )

            if self.use_cmd:  # cmd
                rlist, _, _ = select([sys.stdin], [], [], 0.0)
                if rlist:
                    sys.stdin.readline()
                    self.shell.cmdloop()
                else:
                    pass
