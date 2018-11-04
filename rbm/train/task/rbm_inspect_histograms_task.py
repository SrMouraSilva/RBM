import tensorflow as tf

from rbm.train.task.task import Task


class RBMInspectHistogramsTask(Task):

    def init(self, trainer, session: tf.Session):
        model = trainer.model

        with tf.name_scope('measure/parameters'):
            for parameter in model.parameters:
                tf.summary.histogram(parameter.name, parameter)
