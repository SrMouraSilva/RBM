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
        y = F(visible)
        assert_array_almost_equal(y, y)

    def test_energy(self):
        warnings.warn("Expected a useful test", UserWarning)

        v = T.vector('v')
        h = T.vector('h')
        E = theano.function([v, h], self.rbm.E(v, h))

        visible = self._generate_layer(self.rbm.input_size)
        hidden = self._generate_layer(self.rbm.hidden_size)

        y = E(visible, hidden)
        assert_array_almost_equal(y, y)

    def test_P_h_given_v(self):
        warnings.warn("Expected a useful test", UserWarning)

        v = T.vector('v')

        P_h_given_v = theano.function([v], self.rbm.P_h_given_v(v))

        visible = self._generate_layer(self.rbm.input_size)

        y = P_h_given_v(visible)
        assert_array_almost_equal(y, y)

    def test_P_v_given_h(self):
        warnings.warn("Expected a useful test", UserWarning)

        h = T.vector('h')

        P_v_given_h = theano.function([h], self.rbm.P_v_given_h(h))

        hidden = self._generate_layer(self.rbm.hidden_size)

        y = P_v_given_h(hidden)
        assert_array_almost_equal(y, y)

    def test_sample_h_given_v(self):
        v = T.vector('v')

        sample_h_given_v = theano.function([v], self.rbm.sample_h_given_v(v))

        visible = self._generate_layer(self.rbm.input_size)

        y = [1, 0, 0]
        assert_array_almost_equal(y, sample_h_given_v(visible))

    def test_sample_v_given_h(self):
        h = T.vector('h')

        sample_v_given_h = theano.function([h], self.rbm.sample_v_given_h(h))

        hidden = self._generate_layer(self.rbm.hidden_size)

        y = [0, 1, 1, 0]
        assert_array_almost_equal(y, sample_v_given_h(hidden))

    def test_gibbs_step(self):
        v0 = T.vector('v0')
        gibbs_step = theano.function([v0], self.rbm.gibbs_step(v0))

        visible = self._generate_layer(self.rbm.input_size)

        y = [1, 0, 1, 1]
        assert_array_almost_equal(y, gibbs_step(visible))
