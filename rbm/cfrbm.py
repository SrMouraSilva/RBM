from abc import ABCMeta, abstractmethod

import tensorflow as tf

from rbm.rbm import RBM
from rbm.util.util import softmax, Σ


class CFRBM(RBM):
    """
    RBM for Collaborative Filtering

    :param movies_size: Total of movies
    :param ratings_size: Total of ratings. In the original article
        is 5 (minimum one star and maximum five stars)
    """

    def __init__(self, movies_size: int, ratings_size: int, hidden_size: int, **kwargs):
        self.movie_size = movies_size
        self.rating_size = ratings_size

        super(CFRBM, self).__init__(visible_size=movies_size*ratings_size, hidden_size=hidden_size, **kwargs)

    def setup(self):
        super(CFRBM, self).setup()
        # Call predictions method

    def P_v_given_h(self, h):
        shape_softmax = [-1, self.movie_size, self.rating_size]
        shape_visible = [-1, self.visible_size]

        with tf.name_scope('P_v_given_h'):
            x = h.T @ self.W + self.b_v.T
            x = tf.reshape(x, shape_softmax)

            probabilities = softmax(x)
            return tf.reshape(probabilities, shape_visible).T


class VisibleSamplingMethod(metaclass=ABCMeta):

    def __init__(self):
        self.model: CFRBM = None

    def initialize(self, model: CFRBM):
        """
        :param model: `RBM` model instance
            rbm-like model implemeting :meth:`rbm.model.gibbs_step` method
        """
        self.model = model

    @abstractmethod
    def sample_v_given_h(self, h):
        pass


class ExpectationSamplingMethod(VisibleSamplingMethod):

    def sample(self, probabilities):
        with tf.name_scope('expectation'):
            weights = tf.range(1, self.model.rating_size + 1, dtype=tf.float32)
            expectation = Σ(probabilities * weights, axis=2)

            expectation_rounded = tf.round(expectation)

            x = tf.cast(expectation_rounded, tf.int32)
            return tf.one_hot(x - 1, depth=self.model.rating_size)


class TopKProbabilityElementsMethod(ExpectationSamplingMethod):
    """
    Select the k highest probability elements
    """

    def __init__(self, k: int):
        super(TopKProbabilityElementsMethod, self).__init__()
        self.k = k

    def sample(self, probabilities):
        values, index = tf.nn.top_k(probabilities, k=self.k, sorted=False, name=None)

        result = tf.one_hot(index, depth=self.model.rating_size)
        return tf.reduce_sum(result, axis=2)
