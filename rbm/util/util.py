import numpy as np
from math import e, log as ln
import theano.tensor as T
import theano

#αβχδεφγψιθκλνοπϕστωξυζℂΔΦΓΨΛΣℚℝΞη


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


def Σ(x, axis=1):
    """
    The same of the numpy.sum

    .. math:: \sum_0^{len(x)} x_i
    """
    summation(x, axis)


def summation(x, axis=1):
    """
    The same of the numpy.sum

    .. math:: \sum_0^{len(x)} x_i
    """
    T.sum(x, axis)


def mean(x, axis=0):
    """
    The same of the numpy.mean

    .. math:: \\frac{1}{n} \sum_1^n x_i

    """
    T.mean(x, axis)


def gradient_descent(cost, wrt, consider_constant):
    """
    Gradient descent with automatic differentiation
    https://en.wikipedia.org/wiki/Gradient_descent

    :param cost: (Variable scalar (0-dimensional) tensor variable or None) – Value that we are differentiating (that we want the gradient of). May be None if known_grads is provided.
    :param wrt: (Variable or list of Variables) – Term[s] with respect to which we want gradients
    :param consider_constant: (list of variables) – Expressions not to backpropagate through

    :return: Symbolic expression of gradient of cost with respect to each of the wrt terms. If an element of wrt is not differentiable with respect to the output, then a zero variable is returned.
    """
    gradients = T.grad(cost, wrt, consider_constant=consider_constant)

    return Gradients(gradients, wrt)


class Gradients(object):
    """
    Contains a list of :class:`Gradient`

    :param list[Gradient] gradients: :class:`.Gradient` list
    :param wtr_parameters: Parameters that the gradients realized the automatic differentiation
    """
    def __init__(self, gradients, wrt_parameters):
        self.gradients = [Gradient(gradient, parameter) for gradient, parameter in zip(gradients, wrt_parameters)]

    def __iter__(self):
        return [(gradient, gradient.wrt_parameter) for gradient in self.gradients].__iter__()


class Gradient(object):
    """
    A simple object that contains the gradient with respect to a parameter

    :param expression: The gradient descent generated expression
    :param wrt_parameter: The parameter that are respected to the gradient
    """

    def __init__(self, expression, wrt_parameter):
        self.expression = expression
        self.wrt_parameter = wrt_parameter

    def __mul__(self, other):
        return self.expression * other

    def __rmul__(self, other):
        return self.expression * other


def binomial(n, p, random_state):
    """
    .. math:: X \\sim B(n, p) = Pr(k;n,p)=Pr(X=k)={n \choose k}p^{k}(1-p)^{n-k}

    :param n:
    :param p:
    :param random_state:
    :return:
    """
    return random_state.binomial(size=p.shape, n=n, p=p, dtype=theano.config.floatX)
