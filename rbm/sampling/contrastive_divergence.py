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

    def __str__(self):
        return f'CD-{self.k}'
