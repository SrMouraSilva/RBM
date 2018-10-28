import tensorflow as tf

from rbm.util.util import softmax
from rbm.rbm import RBM


class CFRBM(RBM):

    def __init__(self, users_size: int, movies_size: int, ratings_size: int, hidden_size: int, **kwargs):
        super(RBM, self).__init__(visible_size=movies_size, hidden_size=hidden_size, **kwargs)

        self.rating_size = ratings_size
        self.users_size = users_size

    def sample_v_given_h(self, h):
        with tf.name_scope('sample_v_given_h'):
            probability = self.P_v_given_h(h)
            v_sample = self.sample_prob(probability)

        return v_sample

    def P_v_given_h(self, h):
        mask = tf.sign(self.X)

        x = h.T @ self.W + self.b_v.T

        # Remove all the movie have not been rating
        x = x * mask

        v_prob_tmp = tf.reshape(x,
            [tf.shape(x)[0], -1, self.rating_size]
        )
        v_prob = softmax(v_prob_tmp)

        return tf.reshape(v_prob, [tf.shape(v_prob_tmp)[0], -1])

    @staticmethod
    def sample_prob(probs):
        """
        Takes a tensor of probabilities (as from a sigmoidal activation) and samples from all the distributions
        :param probs: the probabilities tensor
        :return: sampled probabilities tensor
        """
        return tf.nn.relu(tf.sign(probs - tf.random_uniform(tf.shape(probs))))
