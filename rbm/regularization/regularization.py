import tensorflow as tf

from rbm.util.util import Σ


class Regularization(object):
    """
    http://nghiaho.com/?p=1796
    """
    def __init__(self, decay):
        self.decay = tf.constant(decay, dtype=tf.float32, name='decay')
        self.parameter = None

    def initialize(self, parameter):
        self.parameter = parameter

    @property
    def value(self):
        return self.calculate(self.parameter)

    def calculate(self, param):
        raise NameError('Should be implemented by subclasses!')

    def __add__(self, other):
        with tf.name_scope('regularization'):
            return self.value + other

    def __radd__(self, other):
        with tf.name_scope('regularization'):
            return other + self.value

    def __str__(self):
        return self.__class__.__name__


class NoRegularization(Regularization):
    """
    Is no added nothing for the value.

    Design pattern null object implementation
    """
    def __init__(self):
        Regularization.__init__(self, 0.0)
        self.zero = tf.constant(0.0, dtype=tf.float32, name='0')

    def calculate(self, param):
        return self.zero


class L1Regularization(Regularization):
    """
    .. math::

            \\text{L1} = \sum_{i} |\\theta_i|
    """

    def __init__(self, decay):
        Regularization.__init__(self, decay)

    def calculate(self, param):
        with tf.name_scope('L1'):
            return self.decay * Σ(tf.abs(param))


class L2Regularization(Regularization):
    """
    .. math::

            \\text{L2} = 2~decay \cdot \sum_{i} \\theta_i^2
    """

    def __init__(self, decay):
        Regularization.__init__(self, decay)

    def calculate(self, param):
        with tf.name_scope('L2'):
            return 2*self.decay * Σ(param**2)
