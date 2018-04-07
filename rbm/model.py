from abc import abstractmethod, ABCMeta

import numpy as np
from theano.tensor.shared_randomstreams import RandomStreams

from rbm.learning.constant_learning_rate import ConstantLearningRate
from rbm.regularization.regularization import NoRegularization


class Model(metaclass=ABCMeta):
    """
    The RBM extensions inherits it

    :param random_state: Numpy random state
    :param Regularization regularization: L1, L2 or None
    :param LearningRate learning_rate:
    """

    def __init__(self, random_state=None, regularization=None, learning_rate=None):
        self.random_state = np.random.RandomState() if random_state is None else random_state
        self.regularization = NoRegularization() if random_state is None else regularization
        self.learning_rate = ConstantLearningRate(1) if learning_rate is None else learning_rate
        self.theano_random_state = RandomStreams(random_state.randint(2**30))

    @abstractmethod
    def gibbs_step(self, v0):
        """
        Required for sampling methods

        :param v0:

        :return:
        """
        pass
