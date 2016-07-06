from chainer import Chain


class Actor(Chain):

    def __init__(self, _shared, _actor):
        self.is_train = True
        super(Actor, self).__init__(shared=_shared, actor=_actor)

    def __call__(self, _x):
        y = self.shared(_x)
        return self.actor(y)

    def training(self):
        self.is_train = True

    def evaluating(self):
        self.is_train = False


class Critic(Chain):

    def __init__(self, _shared, _critic):
        self.is_train = True
        super(Critic, self).__init__(shared=_shared, critic=_critic)

    def __call__(self, _x):
        y = self.shared(_x)
        return self.critic(y)

    def training(self):
        self.is_train = True

    def evaluating(self):
        self.is_train = False
