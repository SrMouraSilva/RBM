import tensorflow as tf
from tensorflow import square, sqrt

from rbm.learning.learning_rate import LearningRate
from rbm.util.util import parameter_name


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
        self.η = learning_rate
        self.ϵ = epsilon

    def calculate(self, dθ, θ):
        ϵ = self.ϵ
        η = self.η

        with tf.name_scope(f'learning_rate_adagrad_{parameter_name(θ)}'):
            variable = tf.Variable(initial_value=tf.zeros(shape=dθ.shape), name=f'adagrad-{parameter_name(θ)}')

            variable = variable.assign(square(dθ))

            return dθ * η / sqrt(variable + ϵ)

    def __str__(self):
        return f'{self.__class__.__name__}-{self.η}-{self.ϵ}'
