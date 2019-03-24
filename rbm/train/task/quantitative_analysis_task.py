from collections import defaultdict

import tensorflow as tf
from pandas import DataFrame

from experiments.rbm_experiment.rbm_experiment import RBMExperiment
from rbm.train.task.task import Task
from rbm.train.trainer import Trainer
from rbm.util.util import mean


class QuantitativeAnalysisTask(Task):

    def __init__(self, data_train: DataFrame, data_validation: DataFrame, total_movies: int):
        self.data_train = data_train
        self.data_validation = data_validation
        self.total_movies = total_movies

    def init(self, trainer: Trainer, session: tf.Session):
        model = RBMExperiment(trainer.model, self.total_movies)
        _, rating_size = self.data_train.shape

        metrics = defaultdict(list)

        for i in range(self.total_movies):
            metrics['Hit@1'].append(model.accuracy(self.data_validation.values, y_column=i))
            metrics['hit@5'].append(model.hit_ratio(self.data_validation.values, y_column=i, k=5, n_labels=rating_size))
            metrics['MRR'].append(model.mrr(self.data_validation.values, y_column=i))
            metrics['MNDCG'].append(model.mdcg(self.data_validation.values, y_column=i, n_labels=rating_size))
            #metrics['MAP@5'].append(model.map(self.data_validation.values, y_column=i, k=5, n_labels=rating_size, plugins_categories_as_one_hot_encoding=plugins_categories))

        with tf.name_scope(f'mean/quantitative/validation'):
            for k, v in metrics.items():
                tf.summary.scalar(k, mean(v))
