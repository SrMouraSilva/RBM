import tensorflow as tf
from rbm.train.task import Task
from rbm.util.util import prepare_graph


class TensorFlowTask(Task):
    def __init__(self, log=None):
        self.log = log

        self.session = None
        self.writer = None

        self.summary_op = None

        self.trainer = None

    def init(self, trainer, session: tf.Session):
        self.trainer = trainer
        self.session = session

        self.session.run(tf.global_variables_initializer())

        if self.log is not None:
            self.writer = prepare_graph(session, self.log)

        self.summary_op = tf.summary.merge_all()

    def pre_epoch(self, epoch: int):
        pass

    def pre_update(self, epoch, update):
        if epoch % 50 == 0:
            print(epoch)

    def post_update(self, epoch: int, update: int, batch, *args, **kwargs):
        v = self.trainer.v
        summary = self.session.run(self.summary_op, feed_dict={v: batch})

        if self.log is not None:
            self.writer.add_summary(summary, update)

    def post_epoch(self, epoch: int):
        pass

    def finished(self, epoch: int):
        if self.log is not None:
            self.writer.close()
