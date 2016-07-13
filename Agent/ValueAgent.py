class ValueAgent(object):

    def __init__(self):
        pass

    def training(self):
        self.is_train = True

    def evaluating(self):
        self.is_train = False

    def step(self):
        raise Exception()

    def train(self):
        raise Exception()

    def updateTargetQFunc(self):
        raise Exception()

    def save(self, _epoch, _step):
        raise Exception()
