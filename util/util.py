import numpy as np
from math import e, log as ln

#αβχδεφγψιθκλνοπϕστωξυζℂΔΦΓΨΛΣℚℝΞη

def σ(x):
    """
    .. math:: \sigma(x) = \\frac{1}{(1 + e^{-x})}
    """
    return sigmoid(x)


def sigmoid(x):
    """
    .. math:: \sigma(x) = \\frac{1}{(1 + e^{-x})}
    """
    return 1 / (1+(e**-x))


def softplus(x):
    """
    .. math:: soft_{+}(x) = ln(1 + e^x)
    """
    return ln(1 + e**x)


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
    np.sum(x, axis)


def mean(x, axis=0):
    """
    The same of the numpy.mean

    .. math:: \\frac{1}{n} \sum_1^n x_i

    """
    np.sum(x, axis)


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

    return [Gradient(gradient, parameter) for gradient, parameter in zip(gradients, wrt)]


class Gradients(object):
    """
    Contains a list of :class:`Gradient`
    """
    def __init__(self):
        pass


class Gradient(object):
    """
    A simple object that contains the gradient with respect to a parameter

    :param expression: The gradient descent generated expression
    :param wrt_parameter: The parameter that are respected to the gradient
    """

    def __init__(self, expression, wrt_parameter):
        self.expression = expression
        self.wrt_parameter = wrt_parameter
