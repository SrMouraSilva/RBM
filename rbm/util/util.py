import theano
import theano.tensor as T

from theano.tensor.var import Variable

#αβχδεφγψιθκλνοπϕστωξυζℂΔΦΓΨΛΣℚℝΞη

dot = theano.tensor.dot

import warnings
# Add capability of dot multiplication between two variables
# with @ infix syntax
warnings.warn("Probably it will make twice product", UserWarning)
Variable.__matmul__ = lambda self, other: dot(self, other)
Variable.__rmatmul__ = lambda self, other: dot(other, self)


def σ(x):
    """
    .. math:: \sigma(x) = \\frac{1}{(1 + e^{-x})}
    """
    return sigmoid(x)


def sigmoid(x):
    """
    The same as :func:`util.σ`
    """
    return T.nnet.sigmoid(x)


def softplus(x):
    """
    .. math:: soft_{+}(x) = ln(1 + e^x)
    """
    return T.nnet.softplus(x)


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
    return T.sum(x, axis)


def mean(x, axis=None):
    """
    The same of the numpy.mean

    .. math:: \\frac{1}{n} \sum_1^n x_i

    """
    return T.mean(x, axis)


def gradient_descent(cost, wrt, consider_constant=None):
    """
    Gradient descent with automatic differentiation
    https://en.wikipedia.org/wiki/Gradient_descent

    :param cost: (Variable scalar (0-dimensional) tensor variable or None) – Value that we are differentiating (that we want the gradient of). May be None if known_grads is provided.
    :param wrt: (Variable or list of Variables) – Term[s] with respect to which we want gradients
    :param consider_constant: (list of variables) – Expressions not to backpropagate through

    :return: Symbolic expression of gradient of cost with respect to each of the wrt terms. If an element of wrt is not differentiable with respect to the output, then a zero variable is returned.
    """
    return T.grad(cost, wrt, consider_constant=consider_constant)


class Gradient(object):
    """
    A simple object that contains the gradient with respect to a parameter

    :param expression: The gradient descent generated expression
    :param wrt_parameter: The parameter that are respected to the gradient
    """

    def __init__(self, expression, wrt):
        self.expression = expression
        self.wrt = wrt


def binomial(n, p, random_state):
    """
    .. math:: X \\sim B(n, p) = Pr(k;n,p)=Pr(X=k)={n \choose k}p^{k}(1-p)^{n-k}

    :param n:
    :param p:
    :param random_state:
    :return:
    """
    return random_state.binomial(size=p.shape, n=n, p=p, dtype=theano.config.floatX)
