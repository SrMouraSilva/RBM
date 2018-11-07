from typing import List, Tuple

import tensorflow as tf
from tensorflow.contrib.distributions import Bernoulli

# https://en.wikipedia.org/wiki/Mathematical_operators_and_symbols_in_Unicode
#αβχδεφγψιθκλνοπϕστωξυζℂΔΦΓΨΛΣℚℝΞη
#ℂℇℊℋℌℍℎℏ
#ℐℑℒℓℕℙℚℛℜℝ
#ℤΩℨKÅℬℭℯ
#ℰℱℲℳℴℵℶℷℸℹℼℽℾℿ
#ⅅⅆⅇⅈⅉⅎ

tf.Tensor.T = property(lambda self: tf.transpose(self))
tf.Variable.T = property(lambda self: tf.transpose(self))
tf.Variable.__setitem__ = lambda self, x, y: self[x].assign(y)


def σ(x):
    """
    .. math:: \sigma(x) = \\frac{1}{(1 + e^{-x})}
    """
    return sigmoid(x)


def sigmoid(x):
    """
    The same as :func:`~rbm.util.util.σ`
    """
    return tf.sigmoid(x)


def softplus(x):
    """
    .. math:: soft_{+}(x) = ln(1 + e^x)
    """
    return tf.nn.softplus(x)


def softmax(x):
    """
    .. math:: softmax(x) = \\frac{e^x}{\sum_i^j e^x}

    For details, see http://www.faqs.org/faqs/ai-faq/neural-nets/part2/section-12.html
    """
    return tf.nn.softmax(x)


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


def 𝚷(x, axis=None):
    return product_of_sequences(x, axis=axis)


def product_of_sequences(x, axis=None):
    return tf.reduce_prod(x, axis=axis)


def mean(x, axis=None):
    """
    The same of the numpy.mean

    .. math:: \\frac{1}{n} \sum_1^n x_i

    """
    return tf.reduce_mean(x, axis)


def outer(x, y):
    """
    Outer product between x and y

    .. math:: x \\otimes y = xy^T

    :param x:
    :param y:
    :return:
    """
    return tf.einsum('i,j->ij', x, y)


def gradients(cost: tf.Variable, wrt, consider_constant=None) -> List[Tuple['Gradient', tf.Variable]]:
    """
    Gradient with automatic differentiation
    https://en.wikipedia.org/wiki/Gradient_descent

    :param cost: (Variable scalar (0-dimensional) tensor variable or None) – Value that we are differentiating (that we want the gradient of). May be None if known_grads is provided.
    :param wrt: (Variable or list of Variables) – Term[s] with respect to which we want gradients
    :param consider_constant: (list of variables) – Expressions not to backpropagate through

    :return: Symbolic expression of gradient of cost with respect to each of the wrt terms. If an element of wrt is not differentiable with respect to the output, then a zero variable is returned.
    """
    gradients = tf.gradients(cost, wrt, stop_gradients=consider_constant)
    return [(Gradient(dθ, wrt=θ), θ) for dθ, θ in zip(gradients, wrt)]


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


def parameter_name(parameter: tf.Variable):
    """
    :return: Name (not qualified) of the parameter
    """
    return parameter.op.name.split('/')[-1]


def scope_print_values(*args):
    return tf.control_dependencies([tf.print(x) for x in args])


def count_equals(a, b):
    """
    :return: How many elements in (a, b) are equals?
    """
    return tf.reduce_sum(tf.cast(tf.math.equal(a, b), dtype=tf.int32))


def exp(x):
    return tf.exp(x)
