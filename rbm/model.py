from abc import abstractmethod, ABCMeta

import tensorflow as tf

from rbm.learning.constant_learning_rate import ConstantLearningRate
from rbm.regularization.regularization import NoRegularization
from rbm.util.util import parameter_name


class Model(metaclass=ABCMeta):
    """
    The RBM extensions inherits it

    :param Regularization regularization: L1, L2 or None
    :param LearningRate learning_rate:
    """

    def __init__(self, regularization=None, learning_rate=None):
        self.regularization = NoRegularization() if regularization is None else regularization
        self.learning_rate = ConstantLearningRate(1) if learning_rate is None else learning_rate

    @property
    def θ(self):
        return None

    @property
    def parameters(self):
        return self.θ

    @abstractmethod
    def gibbs_step(self, v0):
        """
        Required for sampling methods

        :param v0: :math:`\\boldsymbol{v}^{(0)}` Visible layers

        :return:
        """
        pass

    def learn(self, x, *args, **kwargs) -> [tf.Operation]:
        with tf.name_scope('calculate_parameters'):
            updates = self.calculate_parameters_updates(x, *args, **kwargs)

        assignments = []

        for parameter, update in zip(self.parameters, updates):
            with tf.name_scope(f'assigns/assign_{parameter_name(parameter)}'):
                assignments.append(parameter.assign(update))

        return assignments

    @abstractmethod
    def calculate_parameters_updates(self, x, *args, **kwargs):
        pass
