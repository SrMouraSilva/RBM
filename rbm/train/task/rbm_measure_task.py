import tensorflow as tf

from rbm.sampling.contrastive_divergence import ContrastiveDivergence
from rbm.train.task.rbmcf_measure_task import RBMCFMeasureTask
from rbm.train.trainer import Trainer


class RBMMeasureTask(RBMCFMeasureTask):

    def __init__(self, movies_size, ratings_size, data_train, data_validation):
        super().__init__(data_train, data_validation)
        self.cd = ContrastiveDivergence(k=1)
        self._movie_size = movies_size
        self._rating_size = ratings_size
        self.shape_visibleT = [-1, movies_size*ratings_size]

    def init(self, trainer: Trainer, session: tf.Session):
        self.cd.initialize(trainer.model)
        super().init(trainer, session)

    def predict(self, model, v):
        with tf.name_scope('predict'):
            # Generally, the v already contains the missing data information
            # In this cases, v * mask is unnecessary
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

    @property
    def shape_softmax(self):
        return [-1, self.movie_size, self.rating_size]

    @property
    def rating_size(self):
        return self._rating_size

    @property
    def movie_size(self):
        return self._movie_size
