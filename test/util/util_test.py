import unittest

import numpy as np
from theano import tensor
from theano.tensor.shared_randomstreams import RandomStreams

from rbm.util.util import *


class UtilTest(unittest.TestCase):

    def test_σ(self):
        x = tensor.vector(name='x')
        f = theano.function([x], σ(x))

        x = [0, 2, 8]
        y = [0.5, 0.880797, 0.999665]
        assert np.allclose(y, f(x))

    def test_softplus(self):
        x = tensor.vector(name='x')
        f = theano.function([x], softplus(x))

        x = [1, 2, 8]
        y = [1.31326169, 2.12692801, 8.00033541]
        assert np.allclose(y, f(x))

    def test_Σ(self):
        x = tensor.vector(name='x')
        f = theano.function([x], Σ(x))

        x = [1, 2, 8]
        y = 11
        assert y == f(x)

    def test_mean(self):
        x = tensor.vector(name='x')
        f = theano.function([x], mean(x))

        x = np.asarray([1, 11])
        y = 6
        assert y == f(x)

    def test_gradient_descent(self):
        x = T.dscalar('x')
        z = T.dscalar('z')
        y = x ** 2 + z ** 3

        parameters = [3, 7.5]
        responses = [6, 3 * 7.5**2]

        gradients = gradient_descent(y, [x, z])

        assert responses[0] == theano.function([x, z], gradients[0])(*parameters)
        assert responses[1] == theano.function([x, z], gradients[1])(*parameters)
        assert responses == theano.function([x, z], gradients)(*parameters)

    def test_binomial(self):
        rng = np.random.RandomState()
        theano_rng = RandomStreams(rng.randint(2 ** 30))

        x = tensor.vector(name='x')
        f = theano.function([x], binomial(n=1, p=x, random_state=theano_rng))

        x = [.5]
        y = f(x)
        assert 0 <= y <= 1
