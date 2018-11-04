from abc import ABCMeta
import tensorflow as tf


class LearningRate(metaclass=ABCMeta):
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
        #self.learning_rate = tf.constant(learning_rate, dtype=tf.float32, name='learning_rate')

    @property
    def η(self):
        """
        :return: :attr:`.learning_rate`
        """
        return self.learning_rate

    @property
    def updates(self):
        return None

    def __mul__(self, other):
        return self.learning_rate * other

    def __rmul__(self, other):
        return other * self.learning_rate

    def __str__(self):
        return f'{self.__class__.__name__}-{self.η}'
