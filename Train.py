import Config
import logging


class Train(object):

    def __init__(self, _agent):
        self.agent = _agent
        self.step_count = 0

    def run(self):
        while self.step_count < Config.step_total:
            logging.info('Step: ' + str(self.step_count + 1))
            self.agent.step()
            self.step_count += 1
            if not self.step_count % Config.step_train:
                self.agent.train()
            if not self.step_count % Config.setp_update_target:
                self.agent.updateTargetQFunc()
            if not self.step_count % Config.step_save:
                self.agent.save(self.step_count)
