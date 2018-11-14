import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import sign, sqrt, square

from rbm.rbmcf import RBMCF
from rbm.train.task.task import Task
from rbm.train.trainer import Trainer
from rbm.util.util import mean, Σ, count_equals


class RBMCFMeasureTask(Task):

    def __init__(self, data=None):
        self.model: RBMCF = None
        self.data = data.copy()

    def init(self, trainer: Trainer, session: tf.Session):
        self.model = trainer.model

        data_train = tf.constant(trainer.data_x.T.values, dtype=tf.float32)
        reconstructed = trainer.model.gibbs_step(data_train)

        with tf.name_scope('measure/reconstruction'):
            tf.summary.scalar('hamming', self.hamming_distance(data_train, reconstructed))

        train, test = train_test_split(self.data, test_size=.2, random_state=42)

        values_train = []
        values_test = []

        values_rmse_train = []
        values_rmse_test = []

        for i in range(self.movie_size):
            with tf.name_scope(f'details/measure/evaluate-{i}'):
                value_train, rmse_train = self.evaluate(self.model, train, column=i)
                value_test, rmse_test = self.evaluate(self.model, test, column=i)

                values_train.append(value_train)
                values_test.append(value_test)

                values_rmse_train.append(rmse_train)
                values_rmse_test.append(rmse_test)

                tf.summary.scalar('train', value_train)
                tf.summary.scalar('test', value_test)

                tf.summary.scalar('RMSE_train', rmse_train)
                tf.summary.scalar('RMSE_test', rmse_test)

        with tf.name_scope(f'measure/evaluate'):
            tf.summary.scalar('RMSE_train', mean(values_rmse_train))
            tf.summary.scalar('RMSE_test', mean(values_rmse_test))
            tf.summary.scalar('train', mean(values_train))
            tf.summary.scalar('test', mean(values_test))

    def evaluate(self, model: RBMCF, data, column):
        total_of_elements = data.shape[0]

        i = column * self.rating_size
        j = (column+1) * self.rating_size

        x = data.copy().values
        y = data.copy().values

        x[:, i:j] = 0
        y = y[:, i:j]

        y_predicted = self.predict(model, x.T).T
        y_predicted = y_predicted[:, i:j]

        y_labels = self.argmax(y)
        y_predict_labels = self.argmax(y_predicted)

        total_equals = count_equals(y_labels, y_predict_labels)

        rmse = sqrt(mean(square(y - y_predicted)))

        with tf.name_scope('histogram'):
            tf.summary.histogram('y_label', y_labels)
            tf.summary.histogram('y_predict_label', y_predict_labels)

        return total_equals / total_of_elements, rmse

    def argmax(self, data_y):
        """
        One-hot encoding to categorical
        """
        return tf.argmax(data_y, axis=1)

    def hamming_distance(self, a, b):
        # Element wise binary error
        x = tf.abs(a - b)
        # Reshape to sum all errors by movie
        x = x.reshape(self.shape_softmax)
        # Sum of movie errors
        x = Σ(x, axis=2)
        # If a movie contains two binary erros,
        # we will consider only one error
        x = sign(x)

        # Mean of all errors
        return mean(Σ(x, axis=1))

    def predict(self, model, x):
        return model.predict(x)

    @property
    def shape_softmax(self):
        return self.model.shape_softmax

    @property
    def rating_size(self):
        return self.model.rating_size

    @property
    def movie_size(self):
        return self.model.movie_size
