import logging


class Test(object):

    def __init__(self, _agent):
        self.agent = _agent
        self.agent.evaluating()

    def run(self):
        self.agent.startNewGame()
        while self.agent.env.in_game:
            self.agent.step()
