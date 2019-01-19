from copy import copy
from enum import Enum

import tensorflow as tf
from tensorflow import log

from rbm.rbm import RBM


class BaseRateType(Enum):
    uniform = 1
    b_v = 2
    b_h = 3


class RBMBaseRate:
    def __init__(self, model: RBM):
        self.model = model

    def generate(self, base_type: BaseRateType = BaseRateType.uniform):
        """ base_rate_type = {'uniform', 'c', 'b'} """
        base_rate = copy(self.model)
        base_rate.W = tf.zeros_like(self.model.W)

        annealable_params = []
        compute_lnZ = None

        if base_type == BaseRateType.uniform:
            compute_lnZ, annealable_params = self._uniform(base_rate)

        elif base_type == BaseRateType.b_v:
            compute_lnZ, annealable_params = self._b_v(base_rate)

        elif base_type == BaseRateType.b_h:
            compute_lnZ, annealable_params = self._b_h(base_rate)

        # types.MethodType(compute_lnZ, base_rate)
        base_rate.compute_lnZ = compute_lnZ

        return base_rate, [self.model.W] + annealable_params

    def _uniform(self, base_rate):
        base_rate.b_h = tf.zeros_like(self.model.b_h)
        base_rate.b_v = tf.zeros_like(self.model.b_v)

        def compute_lnZ():
            # Since all parameters are 0, there are 2^hidden_size and 2^input_size
            # different neuron's states having the same energy (i.e. E=1)
            return self.model.visible_size * log(2.) + self.model.hidden_size * log(2.)

        return compute_lnZ, [self.model.b_h, self.model.b_v]

    def _b_v(self, base_rate):
        base_rate.b_h = tf.zeros_like(self.model.b_h)

        def compute_lnZ():
            # Since all parameters are 0 except visible biases, there are 2^hidden_size
            # different hidden neuron's states having the same marginalization over h.
            h = tf.zeros((self.model.hidden_size, 1), dtype=tf.float32)
            lnZ = -self.model.marginalize_over_v(h)
            lnZ += self.model.hidden_size * log(2.)

            return lnZ[0]

        return compute_lnZ, [self.model.b_h]

    def _b_h(self, base_rate):
        base_rate.b_v = tf.zeros_like(self.model.b_v)

        def compute_lnZ():
            # Since all parameters are 0 except hidden biases, there are 2^input_size
            # different visible neuron's states having the same free energy (i.e. marginalization over v).
            v = tf.zeros((1, self.model.visible_size))
            lnZ = -self.model.F(v)
            lnZ += self.model.visible_size * log(2.)

            return lnZ[0]

        return compute_lnZ, [self.model.b_v]
