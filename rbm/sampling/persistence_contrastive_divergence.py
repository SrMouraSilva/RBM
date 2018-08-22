from rbm.sampling.contrastive_divergence import ContrastiveDivergence


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
