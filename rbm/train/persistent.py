from abc import ABCMeta

import tensorflow as tf


class Persistent(metaclass=ABCMeta):

    def save(self, session, path):
        """
        :param session:
        :param path: File that the model will be persisted.
                     Generally a file with extension '.ckpt'
        :return:
        """
        saver = tf.train.Saver()
        return saver.save(session, path)

    def load(self, session: tf.Session, path: str):
        """
        :param session:
        :param path: File that the model will be restored.
                     Generally a file with extension '.ckpt'
        :return:
        """
        tf.train.Saver().restore(session, path)

