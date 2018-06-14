from math import ceil

import numpy as np
import tensorflow as tf
from tensorboard.plugins.beholder import Beholder

from rbm.train.task.task import Task


class BeholderTask(Task):

    def __init__(self, log):
        self.visualizer = None
        self.session = None
        self.log = log

        self.variables = []

    def init(self, trainer, session: tf.Session):
        self.session = session
        self.visualizer = Beholder(logdir=self.log)

        W = trainer.model.W
        b_v = trainer.model.b_v

        self.register(W, [28, 28], columns=33, name='beholder/W_filters')
        self.register(b_v, [28, 28], name='beholder/b_v')

    def register(self, variable, image_shape=None, columns=1, name=None):
        if image_shape is None:
            return self.variables.append(variable)

        total_pixels = np.prod(variable.shape).value

        image_lines, image_columns = image_shape
        lines = ceil(total_pixels / (image_lines * image_columns * columns))
        expected_pixels = (image_columns * image_lines) * columns * lines

        # Complete with zeros if necessary
        var = tf.reshape(variable, [-1])
        var = tf.pad(var, [[0, expected_pixels - total_pixels]])

        # Split variable in n images with shape specified
        var = tf.reshape(var, [-1] + image_shape)
        # Split all images
        var = tf.split(var, num_or_size_splits=columns*lines)
        # Split in lines?
        var = tf.split(var, num_or_size_splits=columns)
        # Concatenate columns
        var = tf.concat(var, axis=3)
        # New shape
        var = tf.reshape(var, [image_lines * lines, image_columns * columns])

        new_variable = tf.Variable(name=name, initial_value=tf.zeros(var.shape), dtype=var.dtype)
        self.variables.append(new_variable.assign(var))

    def post_update(self, epoch: int, update: int, batch, *args, **kwargs):
        updated_values = self.session.run(self.variables)
        self.visualizer.update(self.session, arrays=updated_values, frame=updated_values[0])
