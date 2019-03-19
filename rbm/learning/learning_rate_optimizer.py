from abc import ABCMeta

from rbm.learning.learning_rate import LearningRate


class LearningRateOptimizer(metaclass=ABCMeta, LearningRate):
    """
    Optimizers as learning rate: Only the learning rate calculus, not the param update
    """
    pass
