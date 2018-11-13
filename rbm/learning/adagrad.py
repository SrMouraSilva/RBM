import tensorflow as tf
from tensorflow import square, sqrt

from rbm.learning.learning_rate import LearningRate
from rbm.util.util import Gradient, parameter_name


class ADAGRAD(LearningRate):
    def __init__(self, learning_rate, epsilon=1e-8):
        """
        http://ruder.io/optimizing-gradient-descent/index.html#adagrad

        Implements the ADAGRAD learning rule.

        Parameters
        ----------
        lr: float
            learning rate
        eps: float
            eps needed to avoid division by zero.

        Reference
        ---------
        Duchi, J., Hazan, E., & Singer, Y. (2010).
        Adaptive subgradient methods for online learning and stochastic optimization.
        Journal of Machine Learning
        """
        super().__init__(learning_rate)

        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.parameters = {}

    def __mul__(self, gradient: Gradient):
        learning_rate = self.calculate(gradient.value, gradient.wrt)
        return gradient * learning_rate

    def __rmul__(self, gradient: Gradient):
        return self.__mul__(gradient)

    def calculate(self, dθ, θ):
        with tf.name_scope(f'learning_rate_adagrad_{parameter_name(θ)}'):
            variable = tf.Variable(initial_value=tf.zeros(shape=dθ.shape), name=f'adagrad-{parameter_name(θ)}')
            self.parameters[θ] = variable

            variable = variable.assign(square(dθ))

            return self.η / sqrt(variable + self.epsilon)
