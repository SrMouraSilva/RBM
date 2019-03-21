import tensorflow as tf
from pandas import DataFrame

from rbm.train.task.task import Task
from rbm.train.trainer import Trainer


class QuantitativeAnalysisTask(Task):

    def __init__(self, data_train: DataFrame, data_validation: DataFrame):
        self.data_train = data_train
        self.data_validation = data_validation

    def init(self, trainer: Trainer, session: tf.Session):
        # https://www.tensorflow.org/api_docs/python/tf/py_func
        for i in ?:
            with tf.name_scope(f'quantitative/{i}'):
                tf.summary.scalar('Hit@1', F_train)
                tf.summary.scalar('Hit@5', F_train)
                tf.summary.scalar('MRR', F_train)
                tf.summary.scalar('MNDCG', F_train)
                tf.summary.scalar('Map@5', F_train)

        with tf.name_scope(f'mean/quantitative'):
            tf.summary.scalar('Hit@1', F_train)
            tf.summary.scalar('Hit@5', F_train)
            tf.summary.scalar('MRR', F_train)
            tf.summary.scalar('MNDCG', F_train)
            tf.summary.scalar('Map@5', F_train)