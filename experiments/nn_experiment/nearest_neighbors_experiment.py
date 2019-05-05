import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.neighbors import NearestNeighbors

from experiments.model_evaluate.evaluate_method.evaluate_method_function import HitRatio, MDCG, MAP
from experiments.rbm_experiment.data import Data
from rbm.util.embedding import one_hot_encoding


class KNNData(Data):

    def hide_movies(self, movies: [int]) -> np.ndarray:
        """
        Transform column to missing data
        """
        X = self.data.copy()

        for movie in movies:
            X[:, movie] = -1

        return X

    def extract_movie(self, movie: int) -> np.ndarray:
        return self.data[:, movie]


class NearestNeighborsExperiment:
    def __init__(self, n_labels: int, **kwargs):
        self.n_labels = n_labels
        self.nbrs = NearestNeighbors(**kwargs)
        self.X = None

    def fit(self, X):
        self.X = X
        self.nbrs.fit(X)

    def predict(self, X, y_column):
        return self.predict_proba(X, y_column).argmax(axis=1)

    def predict_proba(self, X, y_column):
        distances, indices = self.nbrs.kneighbors(X)
        y_predict = self.X[indices][:, :, y_column]

        y_predict_one_hot = one_hot_encoding(y_predict, depth=self.n_labels, reshape=False)
        y_predict_one_hot_sum = y_predict_one_hot.sum(axis=1)

        return y_predict_one_hot_sum/self.nbrs.n_neighbors

    def accuracy(self, X: np.ndarray, y_columns: [int]):
        data = KNNData(X, self.n_labels)
        X, ys = data.to_missing_movies(y_columns)

        result = {}

        for y, y_column in zip(ys, y_columns):
            y_pred = self.predict(X, y_column)

            result[y_column] = accuracy_score(y, y_pred, normalize=True)

        return result

    def hit_ratio(self, X: np.ndarray, y_columns: [int], k: int, n_labels: int):
        data = KNNData(X, self.n_labels)
        X, ys = data.to_missing_movies(y_columns)

        result = {}

        for y, y_column in zip(ys, y_columns):
            y = one_hot_encoding(y, depth=self.n_labels, dtype=np.bool)

            y_pred = self.predict_proba(X, y_column)
            result[y_column] = HitRatio.hit_ratio(k, y, y_pred, n_labels)

        return result

    def mdcg(self, X: np.ndarray, y_columns: [int], n_labels: int):
        data = KNNData(X, self.n_labels)
        X, ys = data.to_missing_movies(y_columns)

        result = {}

        for y, y_column in zip(ys, y_columns):
            y = one_hot_encoding(y, depth=self.n_labels, dtype=np.bool)

            y_pred = self.predict_proba(X, y_column)
            result[y_column] = MDCG.mdcg(y, y_pred, n_labels)

        return result

    def map(self, X: np.ndarray, y_columns: [int], k: int, n_labels: int, plugins_categories_as_one_hot_encoding: pd.DataFrame):
        data = KNNData(X, self.n_labels)
        X, ys = data.to_missing_movies(y_columns)

        result = {}

        for y, y_column in zip(ys, y_columns):
            y_pred = self.predict_proba(X, y_column)
            result[y_column] = MAP(k, plugins_categories_as_one_hot_encoding).evaluate_probabilistic(pd.Series(y, name='column'), y_pred, n_labels=n_labels)

        return result
