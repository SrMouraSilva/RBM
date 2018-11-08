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
        super(CFRBM, self).__init__(visible_size=movies_size * ratings_size, hidden_size=hidden_size, **kwargs)

        self.movie_size = movies_size
        self.rating_size = ratings_size

        self.shape_softmax = [-1, self.movie_size, self.rating_size]
        self.shape_visibleT = [-1, self.visible_size]

    def setup(self):
        super(CFRBM, self).setup()
        # Call predictions method
        #  - Expectation
        #  - Top-k

    def P_v_given_h(self, h, mask=None):
        with tf.name_scope('P_v_given_h'):
            x = h.T @ self.W + self.b_v.T

            if mask is not None:
                x = x * mask

            x = tf.reshape(x, self.shape_softmax)

            probabilities = softmax(x)
            return tf.reshape(probabilities, self.shape_visibleT).T

    def predict(self, v, index_missing_movies):
        mask = self.generate_mask(index_missing_movies)
        
        with tf.name_scope('predict'):
            # Generally, the v already contains the missing data information
            # In this cases, v * mask is unnecessary
            p_h = self.P_h_given_v(v * mask)
            p_v = self.P_v_given_h(p_h, mask=mask.T)

            return self.expectation(p_v)

    def generate_mask(self, index_missing_movies):
        ones = tf.ones(shape=[self.movie_size*self.rating_size, 1])
        '''
        ones = tf.Variable(name='a-mask', initial_value=ones, dtype=tf.float32)

        for index in index_missing_movies:
            i = index * self.rating_size
            j = (index+1) * self.rating_size

            ones[i:j] = tf.zeros(self.movie_size, dtype=tf.float32)#.assign()
        '''

        return ones

    def expectation(self, probabilities):
        # The reshape will only works property if the 'probabilities'
        # (that are a vector) are transposed
        probabilities = tf.reshape(probabilities.T, self.shape_softmax)

        with tf.name_scope('expectation'):
            weights = tf.range(1, self.rating_size + 1, dtype=tf.float32)
            expectation = Σ(probabilities * weights, axis=2)

            # FIXME: Use other function instead round
            expectation_rounded = tf.round(expectation)

            x = tf.cast(expectation_rounded, tf.int32)
            return tf.one_hot(x - 1, depth=self.rating_size)


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
    def sample(self, h):
        pass


class TopKProbabilityElementsMethod(VisibleSamplingMethod):
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
