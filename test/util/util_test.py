import unittest

from numpy.testing import assert_array_almost_equal

from rbm.util.util import *
from test.tf_test import TFTest


class UtilTest(TFTest):

    def test_σ(self):
        x = tf.constant([0., 2., 8.], name='x')
        y = [0.5, 0.880797, 0.999665]

        assert_array_almost_equal(y, σ(x).numpy())

    def test_softplus(self):
        x = tf.constant([1., 2., 8.], name='x')
        y = [1.31326169, 2.12692801, 8.00033541]

        assert_array_almost_equal(y, softplus(x).numpy())

    def test_Σ(self):
        x = tf.constant([1, 2, 8], name='x')
        y = 11

        self.assertEqual(y, Σ(x).numpy())

    def test_mean(self):
        x = tf.constant([1, 11.], name='x')
        y = 6

        self.assertEqual(y, mean(x).numpy())

    '''
    def test_gradient(self):
        x = tf.Variable(3., name='x')
        z = tf.Variable(7.5, name='z')
        y = x ** 2 + z ** 3

        responses = [6, 3 * 7.5**2]

        with tf.Session() as session:
            grads = session.run(gradients(y, [x, z]))

        assert_array_almost_equal(responses, grads)
    '''

    def test_bernoulli_sample(self):
        self.restart_seed()

        x = tf.constant([1., 1., 0., 1.], name='x')
        y = [1, 1, 0, 1]

        result = bernoulli_sample(x).numpy()
        assert_array_almost_equal(y, result)
