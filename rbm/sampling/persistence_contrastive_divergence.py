import tensorflow as tf

from rbm.model import Model
from rbm.sampling.contrastive_divergence import ContrastiveDivergence
from rbm.util.util import bernoulli_sample


class PersistentCD(ContrastiveDivergence):

    def __init__(self, k=1, shape=None):
        super(PersistentCD, self).__init__(k=k)

        self.v = None
        self.shape = shape

    def initialize(self, model: Model) -> None:
        super(PersistentCD, self).initialize(model)

        self.v = tf.Variable(name='PCD-v', initial_value=bernoulli_sample(tf.random_uniform(self.shape)))

    def __call__(self, v):
        with tf.name_scope('PCD-{}'.format(self.k)):
            CD = super(PersistentCD, self).__call__

            self.v.assign(CD(self.v))

            return self.v

    def __str__(self):
        return f'PCD-{self.k}'
