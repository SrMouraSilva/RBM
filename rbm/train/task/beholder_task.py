import tensorflow as tf
from tensorboard.plugins.beholder import Beholder

from rbm.train.task.task import Task


class BeholderTask(Task):

    def __init__(self, log):
        self.visualizer = None
        self.session = None
        self.log = log

    def init(self, trainer, session: tf.Session):
        self.session = session
        self.visualizer = Beholder(logdir=self.log)

    def post_update(self, epoch: int, update: int, batch, *args, **kwargs):
        self.visualizer.update(self.session)
