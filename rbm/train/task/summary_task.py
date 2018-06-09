import tensorflow as tf
from rbm.train.task.task import Task
from rbm.util.util import prepare_graph


class SummaryTask(Task):
    def __init__(self, log=None):
        self.log = log

        self.session = None
        self.writer = None

        self.summary_op = None

        self.trainer = None

    def init(self, trainer, session: tf.Session):
        self.trainer = trainer
        self.session = session

        if self.log is not None:
            self.writer = prepare_graph(session, self.log)

        self.summary_op = tf.summary.merge_all()

    def pre_epoch(self, epoch: int):
        if epoch % 2 == 0:
            print('Epoch', epoch)

    def pre_update(self, index: int, batch, *args, **kwargs):
        v = self.trainer.v
        summary = self.session.run(self.summary_op, feed_dict={v: batch})

        if self.log is not None \
        and index % 50 == 0:
            self.writer.add_summary(summary, index)

    def finished(self, epoch: int):
        if self.log is not None:
            self.writer.close()
