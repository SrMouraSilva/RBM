import theano
from theano import tensor

# declare two symbolic floating-point scalars
a = tensor.dscalar()
b = tensor.dscalar()

# create a simple expression
c = a + b

# convert the expression into a callable object that takes (a,b)
# values as input and computes a value for c
f = theano.function([a, b], c)
print(f(2, 3))

##############

import numpy as np
import unittest

from rbm.rbm import RBM
from rbm.sampling.contrastive_divergence import ContrastiveDivergence


rbm = RBM(input_size=4, hidden_size=3, sampling_method=ContrastiveDivergence())
print(rbm)

data_input = np.asarray([0, 0, 0, 0])
print(rbm.F(data_input))
