import tensorflow as tf

from rbm.sampling.contrastive_divergence import ContrastiveDivergence


class PersistentCD(ContrastiveDivergence):

    def __init__(self, k=1):
        super(PersistentCD, self).__init__(k=k)

        self.v = None

    def __call__(self, v):
        if self.v is None:
            self.v = v

        with tf.name_scope('PCD-{}'.format(self.k)):
            self.v = super(PersistentCD, self).__call__(self.v)

            return self.v

    def __str__(self):
        return f'PCD-{self.k}'
