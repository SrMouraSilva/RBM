import numpy as np
import tensorflow as tf

from pandas import DataFrame
from tensorflow import sign, sqrt, square
from tensorflow.python.ops.gen_bitwise_ops import bitwise_and

from rbm.predictor.expectation_predictor import RBMCFExpectationPredictor, RoundMethod, NormalizationRoundingMethod
from rbm.predictor.predictor import Predictor
from rbm.predictor.rbm_top_k_predictor import RBMTop1Predictor, RBMTopKPredictor
from rbm.rbmcf import RBMCF
from rbm.train.task.task import Task
from rbm.train.trainer import Trainer
from rbm.util.util import mean, Σ, count_equals, count_equals_array


class RBMCFMeasureTask(Task):

    def __init__(self, data_train: DataFrame, data_validation: DataFrame):
        self.model: RBMCF = None
        self.data_train = data_train.copy()
        self.data_validation = data_validation.copy()

    def init(self, trainer: Trainer, session: tf.Session):
        self.model = trainer.model

        data_train_numpy = self.data_train.values.T
        reconstructed = self.model.gibbs_step(data_train_numpy)

        with tf.name_scope('measure/reconstruction'):
            tf.summary.scalar('hamming', self.hamming_distance(data_train_numpy, reconstructed))

        for key, predictor in self.predictors.items():
            self.evaluate_predictions(predictor, key)

    def evaluate_predictions(self, predictor: Predictor, identifier: str):
        values_train = []
        values_validation = []

        values_rmse_train = []
        values_rmse_validation = []

        for i in range(self.movie_size):
            with tf.name_scope(f'details/measure/{identifier}/evaluate-{i}'):
                value_train, rmse_train = self.evaluate(predictor, self.data_train, column=i)
                value_validation, rmse_validation = self.evaluate(predictor, self.data_validation, column=i)

                values_train.append(value_train)
                values_validation.append(value_validation)

                values_rmse_train.append(rmse_train)
                values_rmse_validation.append(rmse_validation)

                tf.summary.scalar('train', value_train)
                tf.summary.scalar('validation', value_validation)

                # tf.summary.scalar('RMSE_train', rmse_train)
                # tf.summary.scalar('RMSE_validation', rmse_validation)

        with tf.name_scope(f'measure/evaluate/{identifier}'):
            tf.summary.scalar('RMSE_train', mean(values_rmse_train))
            tf.summary.scalar('RMSE_validation', mean(values_rmse_validation))

            tf.summary.scalar('train', mean(values_train))
            tf.summary.scalar('validation', mean(values_validation))

    def evaluate(self, predictor: Predictor, data, column):
        total_of_elements = data.shape[0]

        i = column * self.rating_size
        j = (column+1) * self.rating_size

        x = data.copy().values
        y = data.copy().values

        x[:, i:j] = 0
        y = y[:, i:j]

        y_predicted = predictor.predict(x.T).T
        y_predicted = y_predicted[:, i:j]

        rmse = sqrt(mean(square(y - y_predicted)))

        if True:
            y = y.astype(np.int32)
            y_predicted = y_predicted.cast(tf.int32)

            total_equals = count_equals_array(bitwise_and(y, y_predicted), y)
        else:
            y_labels = self.argmax(y)
            y_predict_labels = self.argmax(y_predicted)

        #total_equals = count_equals(y_labels, y_predict_labels)

        #with tf.name_scope('histogram'):
        #    tf.summary.histogram('y_label', y_labels)
        #    tf.summary.histogram('y_predict_label', y_predict_labels)

        return total_equals / total_of_elements, rmse

    def argmax(self, data_y):
        """
        One-hot encoding to categorical
        """
        return tf.argmax(data_y, axis=1)

    def hamming_distance(self, a, b):
        # Element wise binary error
        x = tf.abs(a - b)
        # Reshape to sum all errors by movie
        x = x.reshape(self.shape_softmax)
        # Sum of movie errors
        x = Σ(x, axis=2)
        # If a movie contains two binary erros,
        # we will consider only one error
        x = sign(x)

        # Mean of all errors
        return mean(Σ(x, axis=1))

    @property
    def shape_softmax(self):
        return self.model.shape_softmax

    @property
    def rating_size(self):
        return self.model.rating_size

    @property
    def movie_size(self):
        return self.model.movie_size

    @property
    def predictors(self):
        return {
            'top-1': RBMTop1Predictor(self.model, self.movie_size, self.rating_size),
            'top-5': RBMTopKPredictor(self.model, self.movie_size, self.rating_size, k=5),
            'top-50': RBMTopKPredictor(self.model, self.movie_size, self.rating_size, k=50),
            'expectation/round': RBMCFExpectationPredictor(
                self.model, self.movie_size, self.rating_size, normalization=RoundMethod()
            ),
            'expectation/normalized': RBMCFExpectationPredictor(
                self.model, self.movie_size, self.rating_size, normalization=NormalizationRoundingMethod()
            ),
        }
