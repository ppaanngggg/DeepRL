import Config
import logging

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


class Train(object):

    def __init__(self, _agent):
        self.agent = _agent
        self.agent.training()
        self.epoch = 0
        self.step_local = 0
        self.step_count = 0

    def run(self):
        while self.epoch < Config.epoch_total:
            self.agent.startNewGame()
            self.epoch += 1
            self.step_local = 0
            while self.agent.step():
                self.step_local += 1
                self.step_count += 1
                logger.info('Epoch: ' + str(self.epoch) +
                            ' Step: ' + str(self.step_local))
                if not self.step_count % Config.step_train:
                    self.agent.train()
                if not self.step_count % Config.step_update_target:
                    self.agent.updateTargetQFunc()
                if not self.step_count % Config.step_save:
                    self.agent.save(self.step_count)
