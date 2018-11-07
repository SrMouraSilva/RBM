import time
from collections import OrderedDict
from itertools import product
from typing import Dict, Iterable

import numpy as np
import pandas as pd
import tensorflow as tf
from numpy.core.multiarray import dtype

from rbm.cfrbm import CFRBM
from rbm.drbm import DRBM
from rbm.rbm import RBM
from rbm.train.task.persistent_task import PersistentTask
from rbm.train.task.rbm_inspect_histograms_task import RBMInspectHistogramsTask
from rbm.train.task.rbm_inspect_scalars_task import RBMInspectScalarsTask
from rbm.train.task.summary_task import SummaryTask
from rbm.train.task.task import Task
from rbm.train.trainer import Trainer
from rbm.util.util import scope_print_values, count_equals


class Experiment:

    def train(self, cross_validation):
        parameters = self.prepare_parameters(cross_validation)

        for kwargs in parameters:
            tf.reset_default_graph()
            train(**kwargs)

    def prepare_parameters(self, cross_validation) -> Iterable[Dict]:
        cross_validation = OrderedDict(cross_validation)

        create_kwargs = lambda x: {k: v for k, v in zip(cross_validation.keys(), x)}

        return map(create_kwargs, product(*cross_validation.values()))


def train(data_x: pd.DataFrame, data_y: pd.DataFrame, batch_size=10, epochs=100, hidden_size=100, learning_rate=None, regularization=None,
          sampling_method=None, persist=False, model_class=None):
    """
    # Batch_size = 10 or 100
    # https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf
    """

    tf.set_random_seed(42)

    total_elements, size_element = data_x.shape

    if model_class == RBM:
        rbm = RBM(
            visible_size=size_element,
            hidden_size=hidden_size,
            regularization=regularization,
            learning_rate=learning_rate,
            sampling_method=sampling_method,
        )
    elif model_class == DRBM:
        _, target_class_size = data_y.shape

        rbm = DRBM(
            visible_size=size_element,
            hidden_size=hidden_size,
            target_class_size=target_class_size,
            regularization=regularization,
            learning_rate=learning_rate,
            # Is CD ever
            #sampling_method=sampling_method,
        )
    elif model_class == CFRBM:
        rbm = CFRBM(
            movies_size=6,
            ratings_size=int(size_element / 6),
            hidden_size=hidden_size,
            regularization=regularization,
            learning_rate=learning_rate,
            sampling_method=sampling_method,
        )
    else:
        raise Exception('Invalid RBM')

    trainer = Trainer(rbm, data_x, data_y, batch_size=batch_size)

    trainer.stopping_criteria.append(lambda current_epoch: current_epoch > epochs)

    log = f"results/logs/{rbm}/{time.time()}"

    #trainer.tasks.append(RBMInspectScalarsTask())
    #trainer.tasks.append(RBMInspectHistogramsTask())
    if model_class == CFRBM:
        trainer.tasks.append(MeasureTask())
    if model_class == DRBM:
        trainer.tasks.append(MeasureDRBMTask())

    trainer.tasks.append(SummaryTask(log=log))
    # trainer.tasks.append(BeholderTask(log='results/logs'))

    if persist:
        trainer.tasks.append(PersistentTask(path=f"results/model/batch_size={batch_size}/{rbm}/rbm.ckpt"))

    trainer.train()


class MeasureTask(Task):

    def init(self, trainer: Trainer, session: tf.Session):
        model = trainer.model
        data = tf.constant(trainer.data_x.T.values, dtype=tf.float32)

        with tf.name_scope('measure/reconstruction'):
            tf.summary.scalar('suggest-expectation', self.evaluate(model, data))

    def evaluate(self, model: CFRBM, data):
        size = 5 * model.movie_size
        y = data[size:]

        predicted = model.predict(data, index_missing_movies=[5])
        predicted_y = predicted[size:]

        total = count_equals(y, predicted_y)
        return total


class MeasureDRBMTask(Task):

    def init(self, trainer: Trainer, session: tf.Session):
        n_evaluates = 2

        model = trainer.model
        data_x = trainer.data_x.head(n_evaluates).T.values
        data_x = tf.constant(data_x, dtype=tf.float32)

        data_y = trainer.data_y.head(n_evaluates).T.values
        data_y = self.argmax(data_y)

        with tf.name_scope('measure/reconstruction'):
            tf.summary.scalar('suggest-expectation', self.evaluate(model, data_x, data_y))

    def argmax(self, data_y):
        """
        One-hot encoding to categorical
        """
        return tf.argmax(data_y, axis=0)

    def evaluate(self, model: DRBM, x, y):
        D = model.target_class_size

        # Calc the probability of every class
        evaluate = [model.P_y_given_v(i, x) for i in range(D)]
        evaluate = tf.stack(evaluate, axis=0)

        # Discover how the max probability class
        y_predicted = self.argmax(evaluate)

        values = [
            evaluate,
            count_equals(y, y_predicted)
        ]

        with scope_print_values(*values):
            total = count_equals(y, y_predicted)

        return total
