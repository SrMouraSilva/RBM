import unittest
import warnings

import numpy as np
import theano
import theano.tensor as T
from numpy.testing import assert_array_almost_equal
from theano import config

from rbm.rbm import RBM


class RBMTest(unittest.TestCase):

    def _generate_layer(self, size):
        return self.rng.randint(2, size=size).astype(config.floatX)

    @classmethod
    def setup_class(cls):
        cls.rng = np.random.RandomState(42)
        cls.rbm = RBM(input_size=4, hidden_size=3, random_state=cls.rng)

        b_h = cls.rng.randn(cls.rbm.hidden_size).astype(config.floatX)
        b_v = cls.rng.randn(cls.rbm.input_size).astype(config.floatX)

        cls.rbm.b_h.set_value(b_h)
        cls.rbm.b_v.set_value(b_v)

    def test_parameters(self):
        assert self.rbm.parameters == self.rbm.θ

    def test_free_energy(self):
        warnings.warn("Expected a useful test", UserWarning)

        v = T.vector('v')
        F = theano.function([v], self.rbm.F(v))

        visible = self._generate_layer(self.rbm.input_size)
        assert_array_almost_equal(-1.123075, F(visible))

    def test_energy(self):
        warnings.warn("Expected a useful test", UserWarning)

        v = T.vector('v')
        h = T.vector('h')
        E = theano.function([v, h], self.rbm.E(v, h))

        visible = self._generate_layer(self.rbm.input_size)
        hidden = self._generate_layer(self.rbm.hidden_size)

        assert_array_almost_equal(1.692741, E(visible, hidden))

    def test_P_h_given_v(self):
        v = T.vector('v')

        P_h_given_v = theano.function([v], self.rbm.P_h_given_v(v))

        visible = self._generate_layer(self.rbm.input_size)

        y = [0.5648273, 0.12894841, 0.15073547]
        assert_array_almost_equal(y, P_h_given_v(visible))
