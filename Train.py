import logging
import sys
from select import select

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


class Train(object):

    def __init__(self, _agent, _epoch_total=1e4,
                 _step_train=1, _step_update_target=1e3, _step_save=1e3):
        self.agent = _agent
        self.agent.training()
        self.epoch = 0
        self.step_local = 0
        self.step_count = 0

        self.epoch_total = _epoch_total
        self.step_train = _step_train
        self.step_update_target = _step_update_target
        self.step_save = _step_save

    def run(self):
        while self.epoch < self.epoch_total:
            self.agent.startNewGame()
            self.epoch += 1
            self.step_local = 0
            while self.agent.step():
                self.step_local += 1
                self.step_count += 1
                logger.info('Epoch: ' + str(self.epoch) +
                            ' Step: ' + str(self.step_local))
                if not self.step_count % self.step_train:
                    self.agent.train()
                if not self.step_count % self.step_update_target:
                    self.agent.updateTargetQFunc()
                if not self.step_count % self.step_save:
                    self.agent.save(self.epoch, self.step_local)

                rlist, _, _ = select([sys.stdin], [], [], 0.001)
                if rlist:
                    print raw_input('interrupted')
                else:
                    pass
