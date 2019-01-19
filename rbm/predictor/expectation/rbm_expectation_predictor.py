from abc import ABCMeta, abstractmethod

import tensorflow as tf

from rbm.predictor.expectation.expectation_predictor import RoundingMethod
from rbm.predictor.expectation.rbmcf_expectation_predictor import RBMCFExpectationPredictor
from rbm.rbm import RBM
from rbm.util.util import softmax, Σ


class NormalizationPreExpectation(metaclass=ABCMeta):
    """
    Instead the RBMCF that the probabilities are already normalized (because the softmax function),
    is necessary (pre)-normalize the data before calculate the expectation
    """
    @abstractmethod
    def normalize(self, probabilities):
        pass


class ClassicalNormalization(NormalizationPreExpectation):
    def normalize(self, probabilities):
        maximum = Σ(probabilities, axis=2)
        maximum = maximum.reshape(maximum.shape.as_list() + [1])
        return probabilities / maximum


class SoftmaxNormalization(NormalizationPreExpectation):
    def normalize(self, probabilities):
        return softmax(probabilities)


class RBMExpectationPredictor(RBMCFExpectationPredictor):

    def __init__(self, model: RBM, movie_size, rating_size, pre_normalization: NormalizationPreExpectation, normalization: RoundingMethod):
        super().__init__(model, movie_size, rating_size, normalization)
        self.pre_normalization = pre_normalization

    def expectation(self, probabilities):
        # The reshape will only works property if the 'probabilities'
        # (that are a vector) are transposed
        probabilities = probabilities.T.reshape(self.shape_softmax)

        probabilities = self.pre_normalization.normalize(probabilities)

        with tf.name_scope('expectation'):
            weights = tf.range(1, self.rating_size + 1, dtype=tf.float32)
            return Σ(probabilities * weights, axis=2)
