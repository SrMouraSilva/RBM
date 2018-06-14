import tensorflow as tf

from rbm.train.task.task import Task


class RBMMeasureTask(Task):

    def init(self, trainer, session: tf.Session):
        model = trainer.model

        with tf.name_scope('measure/histograms'):
            for parameter in model.parameters:
                tf.summary.histogram(parameter.name, parameter)
