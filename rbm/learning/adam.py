import tensorflow as tf
from tensorflow import sqrt

from rbm.learning.learning_rate_optimizer import LearningRateOptimizer


class Adam(LearningRateOptimizer):
    def __init__(self, alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        https://arxiv.org/pdf/1412.6980.pdf
        :param alpha: Step size (learning rate)
        :param beta1: Exponential decay rate for the moment estimates
        :param beta2: Exponential decay rate for the moment estimates
        """
        self.α = alpha
        self.β1 = beta1
        self.β2 = beta2
        self.ϵ = epsilon

    def calculate(self, dθ):
        α = self.α
        β1 = self.β1
        β2 = self.β2
        ϵ = self.ϵ

        with tf.name_scope(f'adam'):
            t = tf.Variable(initial_value=0, name=f't', dtype=tf.float32)
            m = tf.Variable(initial_value=tf.zeros(shape=dθ.shape), name=f'm')
            v = tf.Variable(initial_value=tf.zeros(shape=dθ.shape), name=f'v')

            t = t.assign(t + 1)

            m = m.assign(β1 * m + (1 - β1) * dθ)
            v = v.assign(β2 * v + (1 - β2) * dθ ** 2)

            m_hat = m / (1 - β1 ** t)
            v_hat = v / (1 - β2 ** t)

            # The subtraction occur outside of the learning rate
            #return θ - α * m_hat / (sqrt(v_hat) + ϵ)
            return α * m_hat / (sqrt(v_hat) + ϵ)

    def __str__(self):
        return f'{self.__class__.__name__}-{self.α}-{self.β1}-{self.β2}-{self.ϵ}'
