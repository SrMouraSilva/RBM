import tensorflow as tf

from rbm.train.task.task import Task
from rbm.train.trainer import Trainer
from rbm.util.util import mean, Σ, parameter_name


class RBMInspectScalarsTask(Task):

    def init(self, trainer: Trainer, session: tf.Session):
        dataset = tf.constant(trainer.dataset.T.values, dtype=tf.float32)
        reconstructed = trainer.model.gibbs_step(dataset)
        model = trainer.model

        with tf.name_scope('measure/reconstruction'):
            #tf.summary.scalar('error', square(mean(tf.abs(dataset - reconstructed))))
            tf.summary.scalar('hamming', self.hamming_distance(dataset, reconstructed))

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
