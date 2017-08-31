import cmd
import logging
import sys
from select import select

from DeepRL.Agent.AgentAbstract import AgentAbstract

logger = logging.getLogger()
logger.setLevel(logging.INFO)


class TrainShell(cmd.Cmd):
    intro = '[[ Welcome to the shell.   Type help or ? to list commands. ]]'
    prompt = '>'

    def __init__(self, _trainer: 'Train'):
        super().__init__()
        self.trainer: Train = _trainer

    def do_save(self, _arg):
        self.trainer.agent.save(
            self.trainer.epoch, self.trainer.step_local
        )

    def do_eval(self, _arg):
        self.trainer.agent.evaluating()
        self.trainer.agent.startNewGame()
        while self.trainer.agent.step():
            pass
        self.trainer.agent.training()

    def do_bye(self, _arg):
        return True


class Train(object):
    def __init__(
            self, _agent: AgentAbstract,
            _epoch_max: int,
            _step_init: int,
            _step_train: int,
            _step_update_target: int,
            _step_save: int,
            _save_path: str='./save',
            _use_cmd: bool=True,
    ):
        """
        one threading trainer

        :param _agent: agent object
        :param _epoch_max: how much games to play
        :param _step_init: how much steps to start train()
        :param _step_train: how much steps between train()
        :param _step_update_target: how much steps between updateTargetFunc()
        :param _step_save: how much steps between save()
        """
        self.agent: AgentAbstract = _agent
        self.agent.training()  # set to training mode

        self.epoch = 0
        self.step_local = 0
        self.step_total = 0

        self.epoch_max = _epoch_max

        self.step_init = _step_init
        self.step_train = _step_train
        self.step_update_target = _step_update_target
        self.step_save = _step_save

        self.save_path = _save_path
        self.use_cmd = _use_cmd
        if self.use_cmd:
            self.shell = TrainShell(self)

    def run(self):
        while self.epoch < self.epoch_max:
            logger.info('Start new game: {}'.format(self.epoch))

            self.agent.startNewGame()
            self.epoch += 1
            self.step_local = 0  # reset local steps

            in_game = True
            while in_game:
                in_game = self.agent.step()
                self.step_local += 1
                self.step_total += 1

                # init finished
                if self.step_total > self.step_init:
                    if not self.step_total % self.step_train:
                        self.agent.train()
                    if not self.step_total % self.step_update_target:
                        self.agent.updateTargetFunc()
                    if not self.step_total % self.step_save:
                        self.agent.save(
                            self.epoch, self.step_local, self.save_path
                        )

                if self.use_cmd:  # cmd
                    rlist, _, _ = select([sys.stdin], [], [], 0.0001)
                    if rlist:
                        sys.stdin.readline()
                        self.shell.cmdloop()
                    else:
                        pass
