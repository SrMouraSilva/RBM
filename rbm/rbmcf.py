import tensorflow as tf

from rbm.rbm import RBM
from rbm.util.util import softmax, Σ


class RBMCF(RBM):
    """
    RBM for Collaborative Filtering

    :param movies_size: Total of movies
    :param ratings_size: Total of ratings. In the original article
        is 5 (minimum one star and maximum five stars)
    """

    def __init__(self, movies_size: int, ratings_size: int, hidden_size: int, **kwargs):
        super().__init__(visible_size=movies_size * ratings_size, hidden_size=hidden_size, **kwargs)

        self.movie_size = movies_size
        self.rating_size = ratings_size

        self.shape_softmax = [-1, self.movie_size, self.rating_size]
        self.shape_visibleT = [-1, self.visible_size]

    def setup(self):
        super().setup()
        # Call predictions method
        #  - Expectation
        #  - Top-k

    def P_v_given_h(self, h, mask=None):
        with tf.name_scope('P_v_given_h'):
            # x is a line vector
            x = h.T @ self.W + self.b_v.T

            if mask is not None:
                x = x * mask

            x = x.reshape(self.shape_softmax)

            probabilities = softmax(x)
            return probabilities.reshape(self.shape_visibleT).T

    def predict(self, v):
        with tf.name_scope('predict'):
            # Generally, the v already contains the missing data information
            # In this cases, v * mask is unnecessary
            p_h = self.P_h_given_v(v)
            p_v = self.P_v_given_h(p_h)

            expectation = self.expectation(p_v)
            expectation_normalized = self.normalize(expectation, self.rating_size)

            one_hot = tf.one_hot(expectation_normalized, depth=self.rating_size)
            return one_hot.reshape(self.shape_visibleT).T

    def expectation(self, probabilities):
        # The reshape will only works property if the 'probabilities'
        # (that are a vector) are transposed
        probabilities = probabilities.T.reshape(self.shape_softmax)

        with tf.name_scope('expectation'):
            weights = tf.range(1, self.rating_size + 1, dtype=tf.float32)
            return Σ(probabilities * weights, axis=2)

    def normalize(self, x, steps):
        """
        Normalize x in number of steps

        :param x: [1 .. steps]
        :param steps

        :return [0 .. steps-1]
        """
        #return tf.round(x) - 1
        numerator = x - 1
        denominator = (steps - 1) / (steps - 10**-8)

        return tf.floor(numerator/denominator).cast(tf.int32)


class TopKProbabilityElementsMethod(object):
    """
    FIXME
    Select the k highest probability elements
    """

    def __init__(self, k: int):
        self.k = k

    def sample(self, probabilities):
        values, index = tf.nn.top_k(probabilities, k=self.k, sorted=False, name=None)

        result = tf.one_hot(index, depth=self.model.rating_size)
        return tf.reduce_sum(result, axis=2)
