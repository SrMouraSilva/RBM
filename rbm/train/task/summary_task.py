import tensorflow as tf
from rbm.train.task.task import Task
from rbm.train.trainer import Trainer
from rbm.util.util import prepare_graph


class SummaryTask(Task):
    def __init__(self, log=None):
        self.log = log

        self.session = None
        self.writer = None

        self.summary_op = None

        self.trainer = None

    def init(self, trainer: Trainer, session: tf.Session):
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

        if self.log is not None \
        and index % 1200 == 0:
            print(f'{index} - Evaluating')
            summary = self.session.run(self.summary_op, feed_dict={v: batch})
            self.writer.add_summary(summary, index)
            print(f'{index} - Evaluated')

    def finished(self, epoch: int):
        if self.log is not None:
            self.writer.close()
