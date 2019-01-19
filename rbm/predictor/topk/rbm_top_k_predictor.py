import tensorflow as tf

from rbm.predictor.predictor import Predictor
from rbm.rbm import RBM
from rbm.util.util import Σ


class RBMTopKPredictor(Predictor):

    def __init__(self, model: RBM, movie_size: int, rating_size: int, k: int):
        super().__init__(model, movie_size, rating_size)
        self.k = k

    def predict(self, v):
        """
        Assign as 1 the k most probable elements. There are the predicted elements
        """
        with tf.name_scope('predict'):
            p_h = self.model.P_h_given_v(v)
            p_v = self.model.P_v_given_h(p_h)

            return self.max_probability(p_v).T

    def max_probability(self, probabilities):
        # The reshape will only works property if the 'probabilities'
        # (that are a vector) are transposed
        probabilities = probabilities.T.reshape(self.shape_softmax)
        values, index = tf.nn.top_k(probabilities, k=self.k, sorted=False)

        result = tf.one_hot(index, depth=self.rating_size)

        result = Σ(result, axis=2)

        return result.reshape(self.shape_visibleT)


class RBMTop1Predictor(RBMTopKPredictor):

    def __init__(self, model: RBM, movie_size: int, rating_size: int):
        super().__init__(model, movie_size, rating_size, k=1)
