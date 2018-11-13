import tensorflow as tf

from rbm.rbm import RBM
from rbm.sampling.contrastive_divergence import ContrastiveDivergence
from rbm.util.util import bernoulli_sample


class PersistentCD(ContrastiveDivergence):

    def __init__(self, k=1, batch_size=1):
        super().__init__(k=k)

        self.v = None
        self.batch_size = batch_size

    def initialize(self, model: RBM) -> None:
        super().initialize(model)

        self.v = None

    def __call__(self, v):
        with tf.name_scope('PCD-{}'.format(self.k)):
            shape = [v.shape[0].value, 1]
            self.v = tf.Variable(name='PCD-v', initial_value=bernoulli_sample(tf.random.uniform(shape)))
            CD = super().__call__

            self.v = self.v.assign(CD(self.v))

            return self.v

    def __str__(self):
        return f'PCD-{self.k}'
