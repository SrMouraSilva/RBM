from typing import Callable

import tensorflow as tf

from rbm.train.trainer import Trainer


class DefinedDecay:
    """
    TODO Change 'Decay' to something more generic
    """

    def __init__(self, values_generator: Callable[[], tf.Tensor]):
        """
        :param values_generator: All possible values in training step based in total of epoch
        """
        self.values_generator = values_generator

    def __mul__(self, other):
        return self.calculate(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def calculate(self, other):
        values = self.values_generator()
        epoch = Trainer.EPOCH_VARIABLE

        return values[epoch] * other

    def __str__(self):
        #return f'{self.__class__.__name__}-{self.values_generator()}'
        return f'{self.__class__.__name__}'
