from abc import ABCMeta, abstractmethod

import tensorflow as tf
from tensorflow import Tensor

from rbm.predictor.predictor import Predictor
from rbm.rbm import RBM
from rbm.util.util import Σ


class RoundingMethod(metaclass=ABCMeta):

    @abstractmethod
    def calculate(self, x, steps: int) -> Tensor:
        """
        After applied the expectation, the domain is [1 ... steps]
        Is necessary to converts to [0 ... steps), because the elements represented
        by one hot encoding starts to index 0.

        :return x rounded
        """
        pass


class RoundMethod(RoundingMethod):

    def calculate(self, x, steps: int) -> Tensor:
        return (tf.round(x) - 1).cast(tf.int32)


class NormalizationRoundingMethod(RoundingMethod):

    def calculate(self, x, steps: int) -> Tensor:
        numerator = x - 1
        denominator = (steps - 1) / (steps - 10 ** -8)

        return tf.floor(numerator / denominator).cast(tf.int32)


class RBMCFExpectationPredictor(Predictor):

    def __init__(self, model: RBM, movie_size, rating_size, normalization: RoundingMethod):
        super().__init__(model, movie_size, rating_size)
        self.rounding = normalization

    def predict(self, v):
        with tf.name_scope('predict'):
            # Generally, the v already contains the missing data information
            # In this cases, v * mask is unnecessary
            p_h = self.model.P_h_given_v(v)
            p_v = self.model.P_v_given_h(p_h)

            expectation = self.expectation(p_v)
            expectation_normalized = self.rounding.calculate(expectation, self.rating_size)

            one_hot = tf.one_hot(expectation_normalized, depth=self.rating_size)
            return one_hot.reshape(self.shape_visibleT).T

    def expectation(self, probabilities):
        # The reshape will only works property if the 'probabilities'
        # (that are a vector) are transposed
        probabilities = probabilities.T.reshape(self.shape_softmax)

        with tf.name_scope('expectation'):
            weights = tf.range(1, self.rating_size + 1, dtype=tf.float32)
            return Σ(probabilities * weights, axis=2)
