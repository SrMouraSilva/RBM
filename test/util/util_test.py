import unittest

import numpy as np
import tensorflow as tf
from numpy.testing import assert_array_almost_equal

from rbm.util.util import *


'''
class UtilTest(unittest.TestCase):

    def test_σ(self):
        x = tf.constant([0., 2., 8.], name='x')
        y = [0.5, 0.880797, 0.999665]

        with tf.Session() as session:
            assert np.allclose(y, session.run(σ(x)))

    def test_softplus(self):
        x = tf.constant([1., 2., 8.], name='x')
        y = [1.31326169, 2.12692801, 8.00033541]

        with tf.Session() as session:
            assert np.allclose(y, session.run(softplus(x)))

    def test_Σ(self):
        x = tf.constant([1, 2, 8], name='x')
        y = 11

        with tf.Session() as session:
            assert y == session.run(Σ(x))

    def test_mean(self):
        x = tf.constant([1, 11.], name='x')
        y = 6

        with tf.Session() as session:
            assert y == session.run(mean(x))

    def test_gradient(self):
        x = tf.Variable(3., name='x')
        z = tf.Variable(7.5, name='z')
        y = x ** 2 + z ** 3

        responses = [6, 3 * 7.5**2]

        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            gradients = session.run(gradient(y, [x, z]))

        assert responses == gradients

    def test_bernoulli_sample(self):
        x = tf.constant([1, 1, 0, 1.], name='x')
        y = [1, 1, 0, 1]
        with tf.Session() as session:
            result = session.run(bernoulli_sample(x))
            assert_array_almost_equal(y, result)
'''