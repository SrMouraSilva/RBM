import tensorflow as tf

from rbm.learning.learning_rate import LearningRate


class AdaMax(LearningRate):
    def __init__(self, alpha=0.002, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        https://arxiv.org/pdf/1412.6980.pdf
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
            u = tf.Variable(initial_value=tf.zeros(shape=dθ.shape), name=f'u')

            t = t.assign(t + 1)

            m = m.assign(β1 * m + (1 - β1) * dθ)
            u = u.assign(tf.maximum(β2 * u, tf.abs(dθ)))

            # The subtraction occur outside of the learning rate
            #return θ - (α / (1 - β1 ** t)) * m / (u + ϵ)
            return (α / (1 - β1 ** t)) * m / (u + ϵ)

    def __str__(self):
        return f'{self.__class__.__name__}-{self.α}-{self.β1}-{self.β2}-{self.ϵ}'
