import numpy as np
import tensorflow as tf

from rbm.train.task.task import Task


class InspectImagesTask(Task):

    def __init__(self):
        self.session = None
        self.model = None
        self.random_element = None

    def init(self, trainer, session: tf.Session):
        self.session = session
        self.model = trainer.model
        self.random_element = np.random.binomial(1, p=[.2]*28**2).astype(np.float32)

        self.summaries()

    def summaries(self):
        dimension = [-1, 28, 28, 1]
        v = tf.reshape(self.random_element, [28**2, 1])
        CD = self.model.sampling_method

        with tf.name_scope('inspect_images_task'):
            image = tf.reshape(self.random_element, dimension)
            tf.summary.image('image/base', image, 1)

            image = tf.reshape(CD(v), dimension)
            tf.summary.image('image/generated/CD-1', image, 1)

            for i in range(1000):
                v = CD(v)

            image = tf.reshape(CD(v), dimension)
            tf.summary.image('image/generated/CD-1000', image, 1)

            image = tf.reshape(self.model.W, dimension)
            tf.summary.image('param/weight', image, 1)

            image = tf.reshape(self.model.b_v, dimension)
            tf.summary.image('param/b_v', image, 1)

            image = tf.reshape(self.model.b_h, dimension)
            tf.summary.image('param/b_h', image, 1)
