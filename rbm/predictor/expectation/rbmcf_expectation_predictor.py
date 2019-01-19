

class RBMCFExpectationPredictor(metaclass=ABCMeta, RBMBaseExpectationPredictior):
    def expectation(self, probabilities):
        # The reshape will only works property if the 'probabilities'
        # (that are a vector) are transposed
        probabilities = probabilities.T.reshape(self.shape_softmax)

        with tf.name_scope('expectation'):
            weights = tf.range(1, self.rating_size + 1, dtype=tf.float32)
            return Σ(probabilities * weights, axis=2)
