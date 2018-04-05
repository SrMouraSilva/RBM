import unittest

import theano
from theano import tensor
from theano.tensor.shared_randomstreams import RandomStreams

from rbm.util.util import *


class UtilTest(unittest.TestCase):

    def test_σ(self):
        x = tensor.vector()
        f = theano.function([x], σ(x))

        x = [0, 2, 8]
        y = [0.5, 0.880797, 0.999665]
        assert np.allclose(y, f(x))

    def test_softplus(self):
        x = tensor.vector()
        f = theano.function([x], softplus(x))

        x = [1, 2, 8]
        y = [1.31326169, 2.12692801, 8.00033541]
        assert np.allclose(y, f(x))

    def atest_Σ(self):
        x = tensor.vector()
        f = theano.function([x], Σ(x, axis=None))

        x = [1, 2, 8]
        y = [11]
        assert np.allclose(y, f(x))

    def test_mean(self):
        x = tensor.vector()
        f = theano.function([x], mean(x))

        x = [1, 11]
        y = [6]
        assert np.allclose(y, f(x))

    def test_gradient_descent(self):
        pass

    def test_binomial(self):
        rng = np.random.RandomState()
        theano_rng = RandomStreams(rng.randint(2 ** 30))

        x = tensor.vector()
        f = theano.function([x], binomial(n=1, p=x, random_state=theano_rng))

        x = [.5]
        y = f(x)
        assert 0 <= y <= 1
