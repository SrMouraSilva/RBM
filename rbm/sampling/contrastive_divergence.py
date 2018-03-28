import numpy as np
import theano
import theano.tensor as T

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
        v_next = v

        for i in range(self.k):
            v_next = self.model.gibbs_step(v)

        # Keep reference of chain_start and chain_end for later use.
        self.chain_start = v
        self.chain_end = v_next

        return self.chain_end, updates


class PersistentCD(ContrastiveDivergence):
    def __init__(self, input_size, nb_particles=128):
        super(PersistentCD, self).__init__()
        self.particles = theano.shared(np.zeros((nb_particles, input_size), dtype=theano.config.floatX))

    def __call__(self, model, chain_start, cdk=1):
        chain_start = self.particles[:chain_start.shape[0]]
        chain_end, updates = ContrastiveDivergence.__call__(self, model, chain_start, cdk)

        # Update particles
        updates[self.particles] = T.set_subtensor(chain_start, chain_end)

        return chain_end, updates
