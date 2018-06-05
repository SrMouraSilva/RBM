import tensorflow as tf
from tensorflow.contrib.distributions import Bernoulli

#αβχδεφγψιθκλνοπϕστωξυζℂΔΦΓΨΛΣℚℝΞη

tf.Tensor.T = property(lambda self: tf.transpose(self))
tf.Variable.T = property(lambda self: tf.transpose(self))


def σ(x):
    """
    .. math:: \sigma(x) = \\frac{1}{(1 + e^{-x})}
    """
    return sigmoid(x)


def sigmoid(x):
    """
    The same as :func:`util.σ`
    """
    return tf.sigmoid(x)


def softplus(x):
    """
    .. math:: soft_{+}(x) = ln(1 + e^x)
    """
    return tf.nn.softplus(x)


def Σ(x, axis=None):
    """
    The same of the numpy.sum

    .. math:: \sum_0^{len(x)} x_i
    """
    return summation(x, axis)


def summation(x, axis=None):
    """
    The same of the numpy.sum

    .. math:: \sum_0^{len(x)} x_i
    """
    return tf.reduce_sum(x, axis)


def mean(x, axis=None):
    """
    The same of the numpy.mean

    .. math:: \\frac{1}{n} \sum_1^n x_i

    """
    return tf.reduce_mean(x, axis)


def gradient(cost, wrt, consider_constant=None):
    """
    Gradient with automatic differentiation
    https://en.wikipedia.org/wiki/Gradient_descent

    :param cost: (Variable scalar (0-dimensional) tensor variable or None) – Value that we are differentiating (that we want the gradient of). May be None if known_grads is provided.
    :param wrt: (Variable or list of Variables) – Term[s] with respect to which we want gradients
    :param consider_constant: (list of variables) – Expressions not to backpropagate through

    :return: Symbolic expression of gradient of cost with respect to each of the wrt terms. If an element of wrt is not differentiable with respect to the output, then a zero variable is returned.
    """
    return tf.gradients(cost, wrt, stop_gradients=consider_constant)


def outer(x, y):
    """
    Outer product between x and y

    .. math:: x \\otimes y = xy^T

    :param x:
    :param y:
    :return:
    """
    return tf.einsum('i,j->ij', x, y)


class Gradient(object):
    """
    A simple object that contains the gradient with respect to a parameter

    :param expression: The gradient descent generated expression
    :param wrt_parameter: The parameter that are respected to the gradient
    """

    def __init__(self, expression, wrt):
        self.expression = expression
        self.wrt = wrt

    def __mul__(self, other):
        return self.expression * other

    def __rmul__(self, other):
        return other * self.expression


def bernoulli(p):
    """
    :math:`X ~ Bernouli(k;p) = p^{k}(1-p)^{1-k}` for :math:`k \in \{0,1\}`

    :param list p: probabilities
    :return:
    """
    return Bernoulli(probs=p)


def bernoulli_sample(p, samples=()):
    """
    Generate samples from a bernoulli distribution (:func:`~rbm.util.bernoulli()`)

    :param list p: probabilities
    :param samples: sample shape
    :return:
    """
    with tf.name_scope('bernoulli_sample'):
        return tf.cast(bernoulli(p).sample(samples), tf.float32)


def square(x: tf.Tensor) -> tf.Tensor:
    """
    Computes square of x element-wise.

    .. math:: y = x * x = x^2

    :param x: Element
    :return:
    """
    return tf.square(x)


def prepare_graph(session: tf.Session, logdir='./graph'):
    return tf.summary.FileWriter(logdir, session.graph)


def save():
    saver = tf.train.Saver()
    pass
