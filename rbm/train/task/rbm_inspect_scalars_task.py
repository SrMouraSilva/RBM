import tensorflow as tf

from rbm.train.task.task import Task
from rbm.train.trainer import Trainer
from rbm.util.util import mean, Σ, parameter_name, square


class RBMInspectScalarsTask(Task):

    def init(self, trainer: Trainer, session: tf.Session):
        data_x = tf.constant(trainer.data_x.T.values, dtype=tf.float32)

        if trainer.data_y is None:
            reconstructed = trainer.model.gibbs_step(data_x)
        else:
            data_y = tf.constant(trainer.data_y.T.values, dtype=tf.float32)
            reconstructed, reconstructed_y = trainer.model.gibbs_step(data_x, data_y)

        model = trainer.model

        with tf.name_scope('measure/reconstruction'):
            #tf.summary.scalar('error', square(mean(tf.abs(data_x - reconstructed))))
            tf.summary.scalar('hamming', self.hamming_distance(data_x, reconstructed))

        with tf.name_scope('measure/activation'):
            total_elements = tf.reduce_sum(reconstructed.T, axis=1)
            reconstructed_mean, reconstructed_std = tf.nn.moments(total_elements, axes=0)

            tf.summary.scalar('min', tf.reduce_min(total_elements))
            tf.summary.scalar('max', tf.reduce_max(total_elements))

            tf.summary.scalar('mean', reconstructed_mean)
            tf.summary.scalar('std', reconstructed_std)

        with tf.name_scope('measure/parameters'):
            for parameter in model.θ:
                tf.summary.scalar(f'{parameter_name(parameter)}/mean', mean(parameter))

        with tf.name_scope('measure/hyperparameters'):
            tf.summary.scalar('regularization', 0 + model.regularization)

        # FIXME - gradients

    def hamming_distance(self, a, b):
        return Σ(tf.abs(a - b))
