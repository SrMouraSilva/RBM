import tensorflow as tf
from rbm.learning.learning_rate import LearningRate


class ConstantLearningRate(LearningRate):

    def __init__(self, learning_rate):
        self.η = learning_rate

    def __str__(self):
        return f'{self.__class__.__name__}-{self.η}'

    def calculate(self, dθ):
        with tf.name_scope(f'constant_learning_rate'):
            return self.η * dθ
