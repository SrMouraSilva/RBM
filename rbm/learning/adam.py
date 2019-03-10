import tensorflow as tf
from tensorflow import sqrt

from rbm.learning.learning_rate import LearningRate
from rbm.util.util import Gradient, parameter_name, scope_print_values


class Adam(LearningRate):
    def __init__(self, alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        https://arxiv.org/pdf/1412.6980.pdf
        """
        super().__init__(None)

        self.α = alpha
        self.β1 = beta1
        self.β2 = beta2
        self.ϵ = epsilon

    def __mul__(self, gradient: Gradient):
        return self.calculate(gradient.value, gradient.wrt)

    def __rmul__(self, gradient: Gradient):
        return self.__mul__(gradient)

    def calculate(self, dθ, θ):
        α = self.α
        β1 = self.β1
        β2 = self.β2
        ϵ = self.ϵ

        with tf.name_scope(f'learning_rate_adam_{parameter_name(θ)}'):
            t = tf.Variable(initial_value=0, name=f't-{parameter_name(θ)}', dtype=tf.float32)
            m = tf.Variable(initial_value=tf.zeros(shape=dθ.shape), name=f'm-{parameter_name(θ)}')
            v = tf.Variable(initial_value=tf.zeros(shape=dθ.shape), name=f'v-{parameter_name(θ)}')

            t = t.assign(t + 1)

            m = m.assign(β1 * m + (1 - β1) * dθ)
            v = v.assign(β2 * v + (1 - β2) * dθ ** 2)

            m_hat = m / (1 - β1 ** t)
            v_hat = v / (1 - β2 ** t)

            # The subtraction occur outside of the learning rate
            #return θ - α * m_hat / (sqrt(v_hat) + ϵ)
            return α * m_hat / (sqrt(v_hat) + ϵ)
