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
        visible_size = int(self.model.b_v.shape[0].value ** (1/2))
        hidden_size = self.model.b_h.shape[0]

        with tf.name_scope('inspect_images_task'):
            input_dimension = [-1, visible_size, visible_size, 1]
            CD = self.model.sampling_method
            v = tf.reshape(self.random_element, [visible_size**2, 1])

            image = tf.reshape(self.random_element, input_dimension)
            tf.summary.image('image/base', image, 1)

            image = tf.reshape(CD(v), input_dimension)
            tf.summary.image('image/generated', image, 1)

            image = tf.reshape(self.model.W, input_dimension)
            tf.summary.image('param/weight', image, 30)

            image = tf.reshape(self.model.W, [-1, visible_size * hidden_size, visible_size, 1])
            tf.summary.image('param/weight_mixed', image, 30)

            image = tf.reshape(self.model.b_v, input_dimension)
            tf.summary.image('param/b_v', image, 1)
