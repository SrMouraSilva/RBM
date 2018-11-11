from itertools import count

import tensorflow as tf

from rbm.rbm import RBM
from rbm.train.batch import Batch
from rbm.train.task.task import Tasks


class Trainer(object):
    """
    Train a RBM Model

    :param Model model:
    :param batch_size:
    :param starting_epoch:
    """

    def __init__(self, model: RBM, data_x, data_y=None, batch_size=1, starting_epoch=0):
        self.model = model
        self.data_x = data_x
        self.data_y = data_y

        self.tasks = Tasks()
        self.stopping_criteria = []

        self.batch = Batch(data_x=data_x, data_y=data_y, start=starting_epoch, size=batch_size)

        self.v = tf.placeholder(shape=[self.model.visible_size, None], name='v', dtype=tf.float32)
        self.y = None

        if data_y is not None:
            total_of_classes = data_y.shape[1]
            self.y = tf.placeholder(shape=[total_of_classes, None], name='v', dtype=tf.float32)

    def train(self):
        learn_op = self.model.learn(self.v, self.y)

        with tf.Session() as session:
            self._train(session, self.v, self.y, learn_op)

    def _train(self, session: tf.Session, v: tf.placeholder, y: tf.placeholder, learn_op: tf.Operation):
        session.run(tf.global_variables_initializer())
        self.tasks.init(self, session)

        epoch = 0
        for epoch in count(step=1):
            if self.stop_now(epoch):
                break

            self.tasks.pre_epoch(epoch)
            for update, (batch_x, batch_y) in enumerate(self.batch):
                data = {v: batch_x}
                if y is not None:
                    data[y] = batch_y

                index = epoch * self.batch.total + update

                self.tasks.pre_update(index, batch_x, epoch, update)

                _ = session.run(learn_op, feed_dict=data)

                self.tasks.post_update(index, batch_x, epoch, update)

            self.tasks.post_epoch(epoch)

        self.tasks.finished(epoch-1)

    def stop_now(self, epoch):
        return any([stopping_criterion(epoch) for stopping_criterion in self.stopping_criteria])
