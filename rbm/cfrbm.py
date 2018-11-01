from abc import ABCMeta, abstractmethod

import tensorflow as tf

from rbm.rbm import RBM
from rbm.util.util import softmax, 𝔼, Σ, bernoulli_sample


class CFRBM(RBM):
    """
    RBM for Collaborative Filtering

    :param movies_size: Total of movies
    :param ratings_size: Total of ratings. In the original article
        is 5 (minimum one star and maximum five stars)
    """

    def __init__(self, movies_size: int, ratings_size: int, hidden_size: int, visible_sampling_method: 'VisibleSamplingMethod', **kwargs):
        self.movie_size = movies_size
        self.rating_size = ratings_size
        self.visible_sampling_method = visible_sampling_method

        super(CFRBM, self).__init__(visible_size=movies_size*ratings_size, hidden_size=hidden_size, **kwargs)

    def setup(self):
        super(CFRBM, self).setup()
        self.visible_sampling_method.initialize(self)

    def sample_v_given_h(self, h):
        with tf.name_scope('sample_v_given_h'):
            return self.visible_sampling_method.sample_v_given_h(h)

    def P_v_given_h(self, h):
        with tf.name_scope('P_h_given_v'):
            return self.visible_sampling_method.P_v_given_h(h)


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

    def P_v_given_h(self, h):
        x = h.T @ self.model.W + self.model.b_v.T

        shape = [-1, self.model.movie_size, self.model.rating_size]
        x = tf.reshape(x, shape)

        #values, index = tf.nn.top_k(softmax(x), k=6, sorted=True, name=None)
        #with tf.control_dependencies([tf.print(values)]):
        return softmax(x)


class ExpectationSamplingMethod(VisibleSamplingMethod):

    def __init__(self):
        super(VisibleSamplingMethod, self).__init__()
        self.ratings_range = None

    def sample_v_given_h(self, h):
        probabilities = self.P_v_given_h(h)
        samples = self.sample(probabilities)

        shape = [-1, self.model.visible_size]
        return tf.reshape(samples, shape).T

    def sample(self, probabilities):
        with tf.name_scope('expectation'):
            weights = tf.range(1, self.model.rating_size + 1, dtype=tf.float32)
            expectation = Σ(probabilities * weights, axis=2)

            expectation_rounded = tf.round(expectation)

            x = tf.cast(expectation_rounded, tf.int32)
            return tf.one_hot(x - 1, depth=self.model.rating_size)


class NotSampleMethod(ExpectationSamplingMethod):

    def sample(self, probabilities):
        return probabilities


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


class RBMLikeMethod(ExpectationSamplingMethod):
    def sample(self, probabilities):
        return bernoulli_sample(p=probabilities)
