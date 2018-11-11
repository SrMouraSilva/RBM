import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import sign

from rbm.cfrbm import CFRBM
from rbm.train.task.task import Task
from rbm.train.trainer import Trainer
from rbm.util.util import mean, Σ, count_equals


class CFRBMInspectScalarsTask(Task):

    def __init__(self):
        self.model: CFRBM = None

    def init(self, trainer: Trainer, session: tf.Session):
        self.model = trainer.model

        data_train = tf.constant(trainer.data_x.T.values, dtype=tf.float32)
        reconstructed = trainer.model.gibbs_step(data_train)

        with tf.name_scope('measure/reconstruction'):
            tf.summary.scalar('hamming', self.hamming_distance(data_train, reconstructed))

        data = self.read_csv('data/pedalboard-plugin-full-bag-of-words.csv')
        train, test = train_test_split(data, test_size=.2, random_state=42)

        values_train = []
        values_test = []

        for i in range(self.model.movie_size):
            with tf.name_scope(f'details/measure/evaluate-{i}'):
                value_train = self.evaluate(self.model, train, column=i)
                value_test = self.evaluate(self.model, test, column=i)

                values_train.append(value_train)
                values_test.append(value_test)

                tf.summary.scalar('train', value_train)
                tf.summary.scalar('test', value_test)

        with tf.name_scope(f'measure/evaluate'):
            tf.summary.scalar('train', mean(values_train))
            tf.summary.scalar('test', mean(values_test))

    def read_csv(self, path):
        return pd.read_csv(path, sep=",", index_col=['index', 'id'])

    def evaluate(self, model: CFRBM, data, column):
        total_of_elements = data.shape[0]

        i = column * model.rating_size
        j = (column+1) * model.rating_size

        x = data.copy().values
        y = data.copy().values

        x[:, i:j] = 0
        y = y[:, i:j]

        y_predicted = model.predict(x.T).T
        y_predicted = y_predicted[:, i:j]

        y_labels = self.argmax(y)
        y_predict_labels = self.argmax(y_predicted)

        total_equals = count_equals(y_labels, y_predict_labels)

        with tf.name_scope('histogram'):
            tf.summary.histogram('y_label', y_labels)
            tf.summary.histogram('y_predict_label', y_predict_labels)

        return total_equals / total_of_elements

    def argmax(self, data_y):
        """
        One-hot encoding to categorical
        """
        return tf.argmax(data_y, axis=1)

    def hamming_distance(self, a, b):
        # Element wise binary error
        x = tf.abs(a - b)
        # Reshape to sum all errors by movie
        x = x.reshape(self.model.shape_softmax)
        # Sum of movie errors
        x = Σ(x, axis=2)
        # If a movie contains two binary erros,
        # we will consider only one error
        x = sign(x)

        # Mean of all errors
        return mean(Σ(x, axis=1))
