from abc import abstractmethod, ABCMeta

import tensorflow as tf

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

        self.θ = None

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

    @abstractmethod
    def learn(self, *args):
        with tf.name_scope('calculate_parameters'):
            updates = self.calculate_parameters_updates(*args)

        assignments = []

        for parameter, update in zip(self.parameters, updates):
            with tf.name_scope(f'assigns/assign_{parameter_name(parameter)}'):
                assignments.append(parameter.assign(update))

        return assignments

    def calculate_parameters_updates(self, *args):
        pass
