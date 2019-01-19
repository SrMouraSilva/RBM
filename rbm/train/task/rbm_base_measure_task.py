from abc import ABCMeta, abstractmethod

import tensorflow as tf
from pandas import DataFrame

from rbm.evaluate.acurracy_evaluate_method import AccuracyEvaluateMethod
from rbm.predictor.predictor import Predictor
from rbm.rbm import RBM
from rbm.train.task.task import Task
from rbm.train.trainer import Trainer
from rbm.util.util import mean, rmse, hamming_distance


class RBMBaseMeasureTask(Task, metaclass=ABCMeta):

    def __init__(self, data_train: DataFrame, data_validation: DataFrame):
        self.model: RBM = None
        self.data_train = data_train.copy()
        self.data_validation = data_validation.copy()

        self._evaluate_method = AccuracyEvaluateMethod()

    def init(self, trainer: Trainer, session: tf.Session):
        self.model = trainer.model

        with tf.name_scope('measure/reconstruction'):
            data_train_numpy = self.data_train.values.T
            reconstructed = self.model.gibbs_step(data_train_numpy)

            tf.summary.scalar('hamming', hamming_distance(data_train_numpy, reconstructed, shape=self.shape_softmax))

        for key, predictor in self.predictors.items():
            self.evaluate_predictions(predictor, key)

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
            tf.summary.scalar('RMSE_train', mean(values_rmse_train))
            tf.summary.scalar('RMSE_validation', mean(values_rmse_validation))

            tf.summary.scalar('train', mean(values_train))
            tf.summary.scalar('validation', mean(values_validation))

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

    @abstractmethod
    @property
    def shape_softmax(self):
        pass

    @abstractmethod
    @property
    def rating_size(self):
        pass

    @abstractmethod
    @property
    def movie_size(self):
        pass

    @abstractmethod
    @property
    def predictors(self):
        return {}
