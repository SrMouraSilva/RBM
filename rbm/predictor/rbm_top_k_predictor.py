import tensorflow as tf

from rbm.predictor.predictor import Predictor


class RBMTop1Predictor(Predictor):

    def predict(self, v):
        with tf.name_scope('predict'):
            p_h = self.model.P_h_given_v(v)
            p_v = self.model.P_v_given_h(p_h)

            return self.max_probability(p_v).T

    def max_probability(self, probabilities):
        # The reshape will only works property if the 'probabilities'
        # (that are a vector) are transposed
        probabilities = probabilities.T.reshape(self.shape_softmax)
        values, index = tf.nn.top_k(probabilities, k=1, sorted=False, name=None)

        result = tf.one_hot(index, depth=self.rating_size)

        return result.reshape(self.shape_visibleT)
