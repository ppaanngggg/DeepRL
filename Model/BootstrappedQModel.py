from chainer import Chain, ChainList


class BootstrappedQModel(ChainList):

    def __init__(self, _shared, _head, _K):
        self.is_train = True
        self.shared = _shared()
        self.head_list = [_head() for _ in range(_K)]
        super(BootstrappedQModel, self).__init__(
            *(self.head_list + [self.shared]))

    def __call__(self, _x):
        y_shared = self.shared(_x, self.is_train)
        y = [head(y_shared, self.is_train) for head in self.head_list]
        return y

    def training(self):
        self.is_train = True

    def evaluating(self):
        self.is_train = False
