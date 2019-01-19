from math import pi

import tensorflow as tf
from tensorflow import sqrt

from rbm.rbm import RBM
from rbm.util.util import exp


class GBRBM(RBM):

    def E(self, v, h):
        pass

    def F(self, v):
        pass

    def P_h_given_v(self, v):
        with tf.name_scope('P_v_given_h'):
            σ = 1

            term = (1 / sqrt(2 * pi)) * (1 / σ)

            numerator = - (h - self.b_h - σ * self.W @ v)
            denominator = (2 * σ**2)

            return term * exp(numerator / denominator)

    def sample_h_given_v(self, v):
        with tf.name_scope('sample_h_given_v'):
            probabilities = self.P_h_given_v(v)
            h_sample = x

            return h_sample