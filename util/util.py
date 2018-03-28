import numpy as np
from math import e, log as ln

#αβχδεφγψιθκλνοπϕστωξυζℂΔΦΓΨΛΣℚℝΞ


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
