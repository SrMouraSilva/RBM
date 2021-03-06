import tensorflow as tf

from rbm.util.util import Σ


class Regularization(object):
    """
    http://nghiaho.com/?p=1796
    """
    def __init__(self, decay):
        self.decay = None
        self._decay_value = decay
        self.model = None

    def initialize(self, model):
        self.decay = tf.constant(self._decay_value, dtype=tf.float32, name='decay')
        self.model = model

    @property
    def value(self):
        return self.calculate(self.model.W)

    def calculate(self, param):
        raise NameError('Should be implemented by subclasses!')

    def __add__(self, other):
        with tf.name_scope('regularization'):
            return self.value + other

    def __radd__(self, other):
        with tf.name_scope('regularization'):
            return other + self.value

    def __str__(self):
        return f'{self.__class__.__name__}-{self._decay_value}'

    def __mul__(self, other):
        with tf.name_scope('regularization'):
            return self.calculate(other) * other

    def __rmul__(self, other):
        with tf.name_scope('regularization'):
            return other * self.calculate(other)


class NoRegularization(Regularization):
    """
    Is no added nothing for the value.

    Design pattern null object implementation
    """
    def __init__(self):
        Regularization.__init__(self, 0.0)
        self.zero = None

    def initialize(self, parameter):
        super().initialize(parameter)
        self.zero = tf.constant(0.0, dtype=tf.float32, name='0')

    def calculate(self, param):
        return self.zero


class L1AutoGradRegularization(Regularization):
    """
    .. math::

            \\text{L1} = \sum_{i} |\\theta_i|
    """

    def __init__(self, decay):
        Regularization.__init__(self, decay)

    def calculate(self, param):
        with tf.name_scope('L1'):
            return self.decay * Σ(tf.abs(param))


class L2AutoGradRegularization(Regularization):
    """
    .. math::

            \\text{L2} = 2~decay \cdot \sum_{i} \\theta_i^2
    """

    def __init__(self, decay):
        Regularization.__init__(self, decay)

    def calculate(self, param):
        with tf.name_scope('L2'):
            return 2*self.decay * Σ(param**2)


class L2(Regularization):
    """
    A Practical Guide to Training Restricted Boltzmann Machines
    Chapter~10. Weight-decay
    """

    def __init__(self, weight_cost):
        Regularization.__init__(self, weight_cost)

    def calculate(self, param):
        with tf.name_scope('constant'):
            return self.decay * param
