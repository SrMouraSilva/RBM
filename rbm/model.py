import numpy as np
from abc import abstractmethod, ABCMeta
from rbm.regularization.regularization import NoRegularization


class Model(metaclass=ABCMeta):
    """
    The RBM extensions inherits it

    :param random_state: Numpy random state
    :param Regularization regularization: L1, L2 or None
    """

    def __init__(self, random_state=None, regularization=None):
        self.random_state = np.random.RandomState() if random_state is None else random_state
        self.regularization = NoRegularization() if random_state is None else regularization

    @abstractmethod
    def gibbs_step(self, v0):
        """
        Required for sampling methods
        :param v0:
        :return:
        """
        pass
