import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score

from experiments.model_evaluate.evaluate_method.evaluate_method_function import HitRatio, MDCG, MAP, MRR
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

    def accuracy(self, X: np.ndarray, y_columns: [int]):
        data = Data(X, self.total_movies)
        X, ys = data.to_missing_movies(y_columns)

        result = {}

        for y, y_column in zip(ys, y_columns):
            y = y.argmax(axis=1)
            y_pred = self.predict(X.T, y_column)

            accuracy = lambda y, y_pred: accuracy_score(y, y_pred, normalize=True)
            result[y_column] = tf.py_func(accuracy, [y, y_pred], np.double)

        return result

    def hit_ratio(self, X: np.ndarray, y_columns: [int], k: int, n_labels: int):
        data = Data(X, self.total_movies)
        X, ys = data.to_missing_movies(y_columns)

        result = {}

        for y, y_column in zip(ys, y_columns):
            y = y.astype(np.bool)

            y_pred = self.predict_proba(X.T, y_column)
            hit_ratio = lambda y, y_pred: HitRatio.hit_ratio(k, y, y_pred, n_labels)

            result[y_column] = tf.py_func(hit_ratio, [y, y_pred], np.double)

        return result

    def mrr(self, X: np.ndarray, y_column: int):
        data = Data(X, self.total_movies)
        X, y = data.to_missing_movie(y_column)

        y = y.astype(np.bool)

        y_pred = self.predict_proba(X.T, y_column)
        mrr = lambda y, y_pred: MRR.mean_reciprocal_rank(y, y_pred)

        return tf.py_func(mrr, [y, y_pred], np.double)

    def mdcg(self, X: np.ndarray, y_columns: [int], n_labels: int):
        data = Data(X, self.total_movies)
        X, ys = data.to_missing_movies(y_columns)

        result = {}

        for y, y_column in zip(ys, y_columns):
            y = y.astype(np.bool)

            y_pred = self.predict_proba(X.T, y_column)
            mdcg = lambda y, y_pred: MDCG.mdcg(y, y_pred, n_labels)

            result[y_column] = tf.py_func(mdcg, [y, y_pred], np.double)

        return result

    def map(self, X: np.ndarray, y_columns: [int], k: int, n_labels: int, plugins_categories_as_one_hot_encoding: pd.DataFrame):
        data = Data(X, self.total_movies)
        X, ys = data.to_missing_movies(y_columns)

        result = {}

        for y, y_column in zip(ys, y_columns):
            y = y.argmax(axis=1)

            y_pred = self.predict_proba(X.T, y_column)
            map = lambda y, y_pred: MAP(k, plugins_categories_as_one_hot_encoding).evaluate_probabilistic(pd.Series(y, name='column'), y_pred, n_labels=n_labels)

            result[y_column] = tf.py_func(map, [y, y_pred], np.double)

        return result
