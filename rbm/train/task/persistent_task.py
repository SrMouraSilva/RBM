from pathlib import Path

import tensorflow as tf

from rbm.train.task.task import Task


class PersistentTask(Task):

    def __init__(self, path: str, force_load=False, save_after_every: int = 500):
        """
        :param path: File that the model will be persisted and restored.
                     Generally a file with extension '.ckpt'
        :param force_load: Force load the model? False will not unload if the
                           model file doesn't exists, i.e., the file will be create
                           when the learning process finish.
        :param save_after_every: Save the model after every n epochs
        """
        self.trainer = None
        self.path = path
        self.session = None
        self.model = None
        self.force_load = force_load
        self.save_after_every = save_after_every

    def init(self, trainer, session: tf.Session):
        self.trainer = trainer
        self.session = session
        self.model = trainer.model

        checkpoint_file = Path(self.path).parent / 'checkpoint'
        if not self.force_load and not checkpoint_file.exists():
            return

        print("Load model from", checkpoint_file)
        self.model.load(session, self.path)

    def post_epoch(self, epoch: int, *args, **kwargs):
        if epoch % self.save_after_every == 0:
            self.model.save(self.session, self.path)

    def finished(self, epoch: int):
        self.model.save(self.session, self.path)
