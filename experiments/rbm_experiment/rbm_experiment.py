import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score

from experiments.model_evaluate.evaluate_method.evaluate_method import HitRatio
from experiments.rbm_experiment.data import Data
from rbm.rbm import RBM


class RBMExperiment:

    def __init__(self, model: RBM, total_movies: int = 6):
        self.model = model
        self.total_movies = total_movies
        self.rating_size = self.model.visible_size / total_movies

    def predict(self, X, y_column):
        p = self.predict_proba(X, y_column)
        return tf.argmax(p, axis=1)

    def predict_proba(self, X, y_column):
        p_h1 = self.model.P_h_given_v(X)
        p_v1 = self.model.P_v_given_h(p_h1)

        return Data(p_v1.T, self.total_movies).extract_movie(y_column)

    def accuracy(self, X: np.ndarray, y_column: int):
        data = Data(X, self.total_movies)
        X, y = data.to_missing_movie(y_column)

        y = y.argmax(axis=1)
        y_pred = self.predict(X.T, y_column)

        accuracy = lambda y, y_pred: accuracy_score(y, y_pred, normalize=True)
        return tf.py_func(accuracy, [y, y_pred], np.double)

    def hit_ratio(self, X: np.ndarray, y_column: int, k, n_labels):
        data = Data(X, self.total_movies)
        X, y = data.to_missing_movie(y_column)

        y = y.astype(np.bool)

        y_pred = self.predict_proba(X.T, y_column)
        hit_ratio = lambda y, y_pred: HitRatio.hit_ratio(k, y, y_pred, n_labels)

        return tf.py_func(hit_ratio, [y, y_pred], np.double)
