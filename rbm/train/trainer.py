from itertools import count

import tensorflow as tf

from rbm.train.batch import Batch
from rbm.train.task import Tasks


class Trainer(object):
    """
    Train a RBM Model

    :param Model model:
    :param dataset:
    :param batch_size:
    :param starting_epoch:
    """

    def __init__(self, model, dataset, batch_size=1, starting_epoch=0):
        self.model = model
        self.dataset = dataset

        self.tasks = Tasks()
        self.stopping_criteria = []

        self.batch = Batch(data=dataset, start=starting_epoch, size=batch_size)

        self.v = tf.placeholder(shape=[self.model.visible_size, None], name='v', dtype=tf.float32)

    def train(self):
        learn_op = self.model.learn(self.v)

        with tf.Session() as session:
            self._train(session, self.v, learn_op)

    def _train(self, session: tf.Session, v: tf.placeholder, learn_op: tf.Operation):
        self.tasks.init(self, session)

        print("train")
        epoch = 0
        for epoch in count(step=1):
            print("época:", epoch)
            if self.stop_now(epoch):
                break

            self.tasks.pre_epoch(epoch)
            for batch in self.batch:
                update = 0
                self.tasks.pre_update(epoch, update)

                y = session.run(learn_op, feed_dict={v: batch})

                self.tasks.post_update(epoch, update, batch)

        self.tasks.finished(epoch-1)

    def stop_now(self, epoch):
        return any([stopping_criterion(epoch) for stopping_criterion in self.stopping_criteria])
