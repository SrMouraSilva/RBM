from abc import ABCMeta, abstractmethod

import numpy as np
import tensorflow as tf
from pandas import DataFrame

from experiments.other_models.utils import one_hot_encoding
from rbm.evaluate.acurracy_evaluate_method import AccuracyEvaluateMethod
from rbm.predictor.predictor import Predictor
from rbm.rbm import RBM
from rbm.train.task.task import Task
from rbm.train.trainer import Trainer
from rbm.util.util import mean, rmse, hamming_distance


class RBMBaseMeasureTask(Task, metaclass=ABCMeta):

    def __init__(self, data_train: DataFrame, data_validation: DataFrame, total_data_noise=1000):
        self.model: RBM = None

        self.total_data_noise = total_data_noise
        self.data_noise = None

        self.data_train = data_train.copy()
        self.data_validation = data_validation.copy()

        self._evaluate_method = AccuracyEvaluateMethod()

    def init(self, trainer: Trainer, session: tf.Session):
        self.model = trainer.model
        self.data_noise = self._generate_data_noise().copy()

        with tf.name_scope('measure/reconstruction'):
            data_train_numpy = self.data_train.values.T
            reconstructed = self.model.gibbs_step(data_train_numpy)

            tf.summary.scalar('hamming', hamming_distance(data_train_numpy, reconstructed, shape=self.shape_softmax))

        for key, predictor in self.predictors.items():
            self.evaluate_predictions(predictor, key)

        with tf.name_scope(f'measure/evaluate/Free'):
            F_train = mean(self.model.F(self.data_train.T.values))
            F_validation = mean(self.model.F(self.data_validation.T.values))
            F_noise = mean(self.model.F(self.data_noise.T.values))

            tf.summary.scalar('mean_free_energy_train', F_train)
            tf.summary.scalar('mean_free_energy_validation', F_validation)

            #tf.summary.scalar('diff_mean_free_energy', F_train - F_validation)
            tf.summary.scalar('ratio_mean_free_energy', F_validation/F_train)
            # https://ieeexplore.ieee.org/document/7783829
            #  Chapter~IV. Eq 17
            tf.summary.scalar('mean_free_energy_gap', F_noise - F_validation)
            tf.summary.scalar('mean_free_energy_noisy', F_noise)

        with tf.name_scope(f'measure/evaluate/reconstruction'):
            reconstruction_train = self.probability_reconstruction(self.data_train.T.values)
            reconstruction_validation = self.probability_reconstruction(self.data_validation.T.values)

            tf.summary.scalar('RMSE_train', rmse(self.data_train.T.values, reconstruction_train))
            tf.summary.scalar('RMSE_validation', rmse(self.data_validation.T.values, reconstruction_validation))

    def probability_reconstruction(self, v):
        with tf.name_scope('predict'):
            p_h = self.model.P_h_given_v(v)
            return self.model.P_v_given_h(p_h)

    def evaluate_predictions(self, predictor: Predictor, identifier: str):
        values_train = []
        values_validation = []

        values_rmse_train = []
        values_rmse_validation = []

        for i in range(self.movie_size):
            with tf.name_scope(f'details/measure/{identifier}/evaluate-{i}'):
                value_train, rmse_train = self.evaluate_predictor(predictor, self.data_train, column=i)
                value_validation, rmse_validation = self.evaluate_predictor(predictor, self.data_validation, column=i)

                values_train.append(value_train)
                values_validation.append(value_validation)

                values_rmse_train.append(rmse_train)
                values_rmse_validation.append(rmse_validation)

                tf.summary.scalar('train', value_train)
                tf.summary.scalar('validation', value_validation)

                # tf.summary.scalar('RMSE_train', rmse_train)
                # tf.summary.scalar('RMSE_validation', rmse_validation)

        with tf.name_scope(f'measure/evaluate/{identifier}'):
            tf.summary.scalar('train', mean(values_train))
            tf.summary.scalar('validation', mean(values_validation))
        #
        #    tf.summary.scalar('RMSE_train_y_predicted', mean(values_rmse_train))
        #    tf.summary.scalar('RMSE_validation_y_predicted', mean(values_rmse_validation))

    def evaluate_predictor(self, predictor: Predictor, data, column):
        i = column * self.rating_size
        j = (column+1) * self.rating_size

        x = data.copy().values
        y = data.copy().values

        x[:, i:j] = 0
        y = y[:, i:j]

        y_predicted = predictor.predict(x.T).T
        y_predicted = y_predicted[:, i:j]

        # with tf.name_scope('histogram'):
        #    tf.summary.histogram('y_label', y_labels)
        #    tf.summary.histogram('y_predict_label', y_predict_labels)

        return self.evaluate(y, y_predicted), rmse(y, y_predicted)

    def evaluate(self, y, y_predicted):
        return self._evaluate_method(y, y_predicted)

    @property
    @abstractmethod
    def shape_softmax(self):
        pass

    @property
    @abstractmethod
    def rating_size(self):
        pass

    @property
    @abstractmethod
    def movie_size(self):
        pass

    @property
    @abstractmethod
    def predictors(self):
        return {}

    def _generate_data_noise(self):
        columns = self.movie_size

        minimum = 0
        maximum = self.rating_size
        size = self.total_data_noise * columns

        values = np.random.randint(minimum, maximum, size=size)
        values = values.reshape([-1, columns])

        values_one_hot = one_hot_encoding(values, maximum)

        return DataFrame(values_one_hot)
