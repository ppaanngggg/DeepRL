import Config
import logging

class Test(object):

    def __init__(self, _agent):
        self.agent = _agent
        self.agent.testing()

    def run(self):
        self.agent.env.startNewGame()
        while self.agent.env.in_game:
            self.agent.step()
