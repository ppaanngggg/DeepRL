from chainer import Chain


class QModel(Chain):

    def __init__(self, _model):
        self.is_train = True
        super(QModel, self).__init__(model=_model)

    def __call__(self, _x):
        return self.model(_x)

    def training(self):
        self.is_train = True

    def evaluating(self):
        self.is_train = False
