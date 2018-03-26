from math import e, log as ln


def sigmoid(x):
    """
    .. math:: \sigma(x) = 1/(1 + e^{-x})
    """
    return 1 / (1+(e**-x))


def softplus(x):
    """
    .. math:: soft_{+}(x) = ln(1 + e^x)
    """
    return ln(1 + e**x)
