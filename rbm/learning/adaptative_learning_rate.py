from typing import Callable

import tensorflow as tf

from rbm.learning.learning_rate import LearningRate
from rbm.train.trainer import Trainer


class AdaptativeLearningRate(LearningRate):

    def __init__(self, values_generator: Callable[[], tf.Tensor]):
        """
        :param values_generator: All possible values in training step based in total of epoch
        """
        self.values_generator = values_generator

    def calculate(self, dθ):
        values = self.values_generator()
        epoch = Trainer.EPOCH_VARIABLE

        return values[epoch] * dθ

    def __str__(self):
        return f'{self.__class__.__name__}-{self.values_generator()}'


class TFLearningRate(LearningRate):

    def __init__(self, method: Callable[[tf.Variable], tf.Tensor]):
        """
        :param values_generator: All possible values in training step based in total of epoch
        """
        self.method = method

    def calculate(self, dθ):
        epoch = Trainer.EPOCH_VARIABLE
        method = self.method(epoch)

        return method * dθ

    def __str__(self):
        epoch = 0
        return f'{self.__class__.__name__}-{self.method(epoch)}'
