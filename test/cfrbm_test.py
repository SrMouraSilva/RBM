import numpy as np
import scipy
from numpy.testing import assert_array_almost_equal

from rbm.cfrbm import CFRBM
from rbm.util.util import softmax
from test.tf_test import TFTest


class CFRBMTest(TFTest):

    def setUp(self):
        super(CFRBMTest, self).setUp()

        self.movie_size = 3
        self.rating_size = 4
        self.hidden_size = 5

        self.rbm = CFRBM(
            movies_size=self.movie_size,
            ratings_size=self.rating_size,
            hidden_size=self.hidden_size
        )

        self.W = np.array(
            range(self.hidden_size * self.movie_size * self.rating_size)
        ).reshape(self.rbm.W.shape)
        self.W = self.W * .01

        self.b_h = np.array(range(self.rbm.hidden_size)).reshape(self.rbm.b_h.shape)
        self.b_v = np.array(range(self.rbm.visible_size)).reshape(self.rbm.b_v.shape)

        self.rbm.W.assign(self.W)
        self.rbm.b_h.assign(self.b_h)
        self.rbm.b_v.assign(self.b_v)

        self.size_minibatch = 2

        self.visible_shape = [self.rbm.visible_size, self.size_minibatch]
        self.hidden_shape = [self.rbm.hidden_size, self.size_minibatch]

        self.v = self._create_v()
        self.h = self._create_h()

    def _create_v(self):
        self.restart_seed()
        v = np.random.randint(2, size=self.rbm.visible_size * self.size_minibatch)
        return v.reshape(self.visible_shape)

    def _create_h(self):
        self.restart_seed()
        h = np.random.randint(2, size=self.rbm.hidden_size * self.size_minibatch)
        return h.reshape(self.hidden_shape)

    def test__init__(self):
        self.assertEqual(self.movie_size, self.rbm.movie_size)
        self.assertEqual(self.rating_size, self.rbm.rating_size)
        self.assertEqual(self.movie_size*self.rating_size, self.rbm.visible_size)
        self.assertEqual(self.hidden_size, self.rbm.hidden_size)

        assert_array_almost_equal(self.W, self.rbm.W.numpy())
        assert_array_almost_equal(self.b_h, self.rbm.b_h.numpy())
        assert_array_almost_equal(self.b_v, self.rbm.b_v.numpy())

        self.assertEqual([-1, self.rbm.visible_size], self.rbm.shape_visibleT)
        self.assertEqual([-1, self.movie_size, self.rating_size], self.rbm.shape_softmax)

    def test_P_h_given_v(self):
        x = self.W @ self.v + self.b_h
        p_expected = scipy.special.expit(x)
        p_rbm = self.rbm.P_h_given_v(self.v)

        self.assertEquals(self.hidden_shape, p_rbm.shape)
        assert_array_almost_equal(p_expected, p_rbm.numpy())

    def test_P_v_given_h(self):
        x = self.W.T @ self.h + self.b_v
        x = x.T.reshape(self.rbm.shape_softmax)

        y_expected = softmax(x).reshape(self.rbm.shape_visibleT)
        y_expected = y_expected.T

        y = self.rbm.P_v_given_h(self.h)

        assert_array_almost_equal(y_expected, y)

    def test_expectation(self):
        def expectation(x):
            weights = np.array(range(1, self.rbm.rating_size+1), dtype=np.float32)
            x = x.T.reshape(self.rbm.shape_softmax)
            x = x * weights
            return x.sum(axis=2)

        x = np.array([range(self.rbm.visible_size)]).T

        assert_array_almost_equal(
            expectation(x),
            self.rbm.expectation(x).numpy()
        )

        p = self.rbm.P_v_given_h(self.h).numpy()
        assert_array_almost_equal(
            expectation(p),
            self.rbm.expectation(p).numpy()
        )

    def test_normalize(self):
        y = self.rbm.normalize(np.array([1, 1.4, 1.5, 1.51, 1.6, 2]), 2)
        assert_array_almost_equal([0, 0, 0, 1, 1, 1], y.numpy())
