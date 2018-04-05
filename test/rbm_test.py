import unittest

import numpy as np

from rbm.rbm import RBM
from rbm.sampling.contrastive_divergence import ContrastiveDivergence


class RBMTest(unittest.TestCase):

    @classmethod
    def setup_class(cls):
        cls.rbm = RBM(input_size=4, hidden_size=3, sampling_method=ContrastiveDivergence())

    def test_free_energy(self):
        F = lambda visible: self.rbm.F(visible)

        visible = np.asarray([0, 0, 0, 0])
        #np.testing.assert_array_equal(visible, F(visible))
