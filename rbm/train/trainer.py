from itertools import count

import tensorflow as tf

from rbm.rbm import RBM
from rbm.train.shuffled_dataset import ShuffledDataset
from rbm.train.task.task import Tasks


class Trainer(object):
    """
    Train a RBM Model

    :param Model model:
    :param batch_size:
    """

    EPOCH_VARIABLE = None

    @staticmethod
    def get_epoch():
        return Trainer.EPOCH_VARIABLE

    def __init__(self, model: RBM, data, batch_size=1):
        self.model = model
        self.data = data

        self.tasks = Tasks()
        self.stopping_criteria = []

        self.dataset = ShuffledDataset(data=data, batch_size=batch_size)

    def train(self):
        v = self.dataset.get_next()
        with tf.name_scope('train'):
            Trainer.EPOCH_VARIABLE = tf.placeholder(name='epoch', shape=tuple(), dtype=tf.int32)

        learn_op = self.model.learn(v)
        operations = learn_op

        with tf.Session() as session:
            self._train(session, operations)

    def _train(self, session: tf.Session, operations: [tf.Operation]):
        session.run(tf.global_variables_initializer())
        self.tasks.init(self, session)

        epoch = 0
        for epoch in count(step=1):
            if self.is_stop_now(epoch):
                break

            self.tasks.pre_epoch(epoch)
            for update in self.dataset:
                index = epoch * self.dataset.total_batches + update

                self.tasks.pre_update(index, epoch, update)

                _ = session.run(operations, feed_dict={Trainer.EPOCH_VARIABLE: epoch})

                self.tasks.post_update(index, epoch, update)

            self.tasks.post_epoch(epoch)

        self.tasks.finished(epoch-1)

    def is_stop_now(self, epoch):
        return any([stopping_criterion(epoch) for stopping_criterion in self.stopping_criteria])
