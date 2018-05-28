import numpy as np
import tensorflow as tf

from rbm.sampling.sampling_method import SamplingMethod


class ContrastiveDivergence(SamplingMethod):
    """
    :param int k: Number of Gibbs step to do
    """

    def __init__(self, k=1):
        super(ContrastiveDivergence, self).__init__()

        self.chain_start = None
        self.chain_end = None
        self.k = k

    def __call__(self, v):
        """
        :param v: Visible layer
        :return:
        """
        with tf.name_scope('CD-{}'.format(self.k)):
            v_next = v

            for i in range(self.k):
                v_next = self.model.gibbs_step(v_next)

            # Keep reference of chain_start and chain_end for later use.
            self.chain_start = v
            self.chain_end = v_next

            return self.chain_end


class PersistentCD(ContrastiveDivergence):
    def __init__(self, k=1, nb_particles=128):
        super(PersistentCD, self).__init__(k=k)
        self.particles = theano.shared(np.zeros((nb_particles, self.model.visible_size), dtype=tf.float32))

    def __call__(self, chain_start):
        with tf.name_scope('PCD-={}'.format(self.k)):
            chain_start = self.particles[:chain_start.shape[0]]
            chain_end, updates = ContrastiveDivergence.__call__(self, chain_start)

            # Update particles
            updates[self.particles] = T.set_subtensor(chain_start, chain_end)

            return chain_end, updates
