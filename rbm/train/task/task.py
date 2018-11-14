from abc import ABCMeta
import tensorflow as tf


class Task(metaclass=ABCMeta):

    def init(self, trainer, session: tf.Session):
        pass

    def pre_epoch(self, epoch: int):
        pass

    def pre_update(self, index: int, epoch: int, update: int):
        pass

    def post_update(self, index: int, epoch: int, update: int):
        pass

    def post_epoch(self, epoch: int):
        pass

    def finished(self, epoch: int):
        pass


class Tasks(Task):
    def __init__(self):
        super().__init__()
        self._tasks = []

    def append(self, task: Task):
        self._tasks.append(task)

    def init(self, trainer, session: tf.Session):
        for task in self._tasks:
            task.init(trainer, session)

    def pre_epoch(self, epoch):
        for task in self._tasks:
            task.pre_epoch(epoch)

    def pre_update(self, *args, **kwargs):
        for task in self._tasks:
            task.pre_update(*args, **kwargs)

    def post_update(self, *args, **kwargs):
        for task in self._tasks:
            task.post_update(*args, **kwargs)

    def post_epoch(self, epoch):
        for task in self._tasks:
            task.post_epoch(epoch)

    def finished(self, epoch):
        for task in self._tasks:
            task.finished(epoch)
