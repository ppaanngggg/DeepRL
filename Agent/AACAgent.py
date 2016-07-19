from ..Model.ACModel import Actor, Critic
from Agent import Agent
import random
from chainer import serializers, Variable
import chainer.functions as F
import numpy as np

import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


class AACAgent(Agent):

    def __init__(self, _shared, _actor, _critic, _env, _is_train=True,
                _act)
