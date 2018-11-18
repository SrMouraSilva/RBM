import tensorflow as tf
from abc import ABCMeta, abstractmethod

from rbm.rbm import RBM
from rbm.util.util import Σ


class Predictor(metaclass=ABCMeta):

    def __init__(self, model: RBM, movie_size: int, rating_size: int):
        self.model = model

        self.movie_size = movie_size
        self.rating_size = rating_size

        self.shape_softmax = [-1, movie_size, rating_size]
        self.shape_visibleT = [-1, model.visible_size]

    @abstractmethod
    def predict(self, v):
        pass

'''
class RBMExpectationPredictor(RBMCFExpectationPredictor):

    def predict(self, v):
        with tf.name_scope('predict'):
            # Generally, the v already contains the missing data information
            # In this cases, v * mask is unnecessary
            p_h = self.model.P_h_given_v(v)
            p_v = self.model.P_v_given_h(p_h)

            #return self.max_probability(p_v).T

            #expectation = self.softmax_expectation(p_v).T
            expectation = self.normalized_expectation(p_v).T
            expectation_normalized = self.normalize(expectation, self.rating_size)

            one_hot = tf.one_hot(expectation_normalized, depth=self.rating_size)
            return one_hot.reshape(self.shape_visibleT).T

    def normalized_expectation(self, probabilities):
        # The reshape will only works property if the 'probabilities'
        # (that are a vector) are transposed
        probabilities = probabilities.T.reshape(self.shape_softmax)
        maximum = Σ(probabilities, axis=2)
        maximum = maximum.reshape(maximum.shape.as_list() + [1])
        probabilities = probabilities/maximum

        with tf.name_scope('expectation'):
            weights = tf.range(1, self.rating_size + 1, dtype=tf.float32)
            return Σ(probabilities * weights, axis=2)

    def softmax_expectation(self, probabilities):
        # The reshape will only works property if the 'probabilities'
        # (that are a vector) are transposed
        probabilities = probabilities.T.reshape(self.shape_softmax)
        probabilities = softmax(probabilities)

        with tf.name_scope('expectation'):
            weights = tf.range(1, self.rating_size + 1, dtype=tf.float32)
            return Σ(probabilities * weights, axis=2)
'''
