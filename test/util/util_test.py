import unittest

import numpy as np
import tensorflow as tf
from numpy.testing import assert_array_almost_equal

from rbm.util.util import *


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

    '''
    def test_mean(self):
        x = tensor.vector(name='x')
        f = theano.function([x], mean(x))

        x = np.asarray([1, 11])
        y = 6
        assert y == f(x)

    def test_gradient(self):
        x = T.dscalar('x')
        z = T.dscalar('z')
        y = x ** 2 + z ** 3

        parameters = [3, 7.5]
        responses = [6, 3 * 7.5**2]

        gradients = gradient(y, [x, z])

        assert responses[0] == theano.function([x, z], gradients[0])(*parameters)
        assert responses[1] == theano.function([x, z], gradients[1])(*parameters)
        assert responses == theano.function([x, z], gradients)(*parameters)
    '''

    def test_bernoulli(self):
        x = tf.constant([1, 1, 0, 1.], name='x')
        y = [1, 1, 0, 1]
        with tf.Session() as session:
            result = session.run(bernoulli(x).sample())
            assert_array_almost_equal(y, result)
