from abc import abstractmethod, ABCMeta


from rbm.learning.constant_learning_rate import ConstantLearningRate
from rbm.regularization.regularization import NoRegularization


class Model(metaclass=ABCMeta):
    """
    The RBM extensions inherits it

    :param Regularization regularization: L1, L2 or None
    :param LearningRate learning_rate:
    """

    def __init__(self, regularization=None, learning_rate=None):
        self.regularization = NoRegularization() if regularization is None else regularization
        self.learning_rate = ConstantLearningRate(1) if learning_rate is None else learning_rate

    @abstractmethod
    def gibbs_step(self, v0):
        """
        Required for sampling methods

        :param v0:

        :return:
        """
        pass
