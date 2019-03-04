import tensorflow as tf

from rbm.rbm import RBM
from rbm.util.util import softmax


class RBMCF(RBM):
    """
    Restricted Bolzmann Machine for Collaborative Filtering

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

    def P_v_given_h(self, h):
        with tf.name_scope('P_v_given_h'):
            # x is a line vector
            x = h.T @ self.W + self.b_v.T

            x = x.reshape(self.shape_softmax)

            probabilities = softmax(x)
            return probabilities.reshape(self.shape_visibleT).T
