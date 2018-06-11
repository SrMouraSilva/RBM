import tensorflow as tf

from rbm.util.util import Σ


class Regularization(object):
    def __init__(self, decay):
        self.decay = tf.constant(decay, dtype=tf.float32, name='decay')
        self.parameter = None

    def initialize(self, parameter):
        self.parameter = parameter

    @property
    def value(self):
        return self(self.parameter)

    def __call__(self, param):
        raise NameError('Should be implemented by subclasses!')

    def __add__(self, other):
        return self.value + other

    def __radd__(self, other):
        return other + self.value


class NoRegularization(Regularization):
    def __init__(self):
        Regularization.__init__(self, 0.0)
        self.zero = tf.constant(0.0, dtype=tf.float32, name='0')

    def __call__(self, param):
        return self.zero


class L1Regularization(Regularization):
    def __init__(self, decay):
        Regularization.__init__(self, decay)

    def __call__(self, param):
        with tf.name_scope('L1'):
            return self.decay * Σ(tf.abs(param))


class L2Regularization(Regularization):
    def __init__(self, decay):
        Regularization.__init__(self, decay)

    def __call__(self, param):
        with tf.name_scope('L2'):
            return 2*self.decay * Σ(param**2)
