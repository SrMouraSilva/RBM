import tensorflow as tf
from rbm.train.task.task import Task
from rbm.train.trainer import Trainer
from rbm.util.util import prepare_graph


class SummaryTask(Task):
    def __init__(self, log=None, epoch_step=5, every_epoch=None):
        self.log = log

        self.session = None
        self.writer = None

        self.summary_op = None

        self.trainer = None
        self.epoch_step = epoch_step
        self.every_epoch = every_epoch

    def init(self, trainer: Trainer, session: tf.Session):
        self.trainer = trainer
        self.session = session

        if self.log is not None:
            self.writer = prepare_graph(session, self.log)

        self.summary_op = tf.summary.merge_all()

    def pre_epoch(self, epoch: int):
        if epoch % self.epoch_step == 0:
            print('Epoch', epoch)

    def post_epoch(self, epoch: int):
        every_epoch = self.every_epoch is not None and epoch <= self.every_epoch
        epoch_step = epoch % self.epoch_step == 0

        if every_epoch or epoch_step:
            self.evaluate(epoch+1)

    def finished(self, epoch: int):
        if self.log is not None:
            self.writer.close()

    def evaluate(self, index):
        print(f'{index} - Evaluating')
        summary = self.session.run(self.summary_op)
        self.writer.add_summary(summary, index)
        print(f'{index} - Evaluated')
