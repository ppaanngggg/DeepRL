import numpy as np
import chainer.functions as F
import chainer.links as L
from chainer import Variable
from chainer import optimizers
from chainer import Chain, ChainList
import chainer.serializers as S
import chainer.computational_graph as c
import Config


class Shared(Chain):

    def __init__(self):
        super(Shared, self).__init__(

        )

    def __call__(self, _x, _is_train):
        y = _x
        return y


class Head(Chain):

    def __init__(self):
        super(Head, self).__init__(

        )

    def __call__(self, _x, _is_train):
        y = _x
        return y


class Model(ChainList):

    def __init__(self, _shared, _head):
        self.is_train = True
        self.shared = _shared()
        self.head_list = [_head() for _ in range(Config.K)]

        super(Model, self).__init__(*(self.head_list + [self.shared]))

    def __call__(self, _x):
        y_shared = self.shared(_x, self.is_train)
        y = [head(y_shared, self.is_train) for head in self.head_list]
        return y

    def training(self):
        self.is_train = True

    def evaluating(self):
        self.is_train = False


def buildModel(_shared, _head, _pre_model=None):
    q_func = Model(_shared, _head)
    if _pre_model:
        S.load_npz(_pre_model, q_func)
    target_q_func = q_func.copy()
    return q_func, target_q_func
