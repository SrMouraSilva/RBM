import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from experiments.model_evaluate.evaluate_method.evaluate_method_function import HitRatio, MDCG, MAP
from experiments.multiple_positions_experiment.nearest_neighbors_experiment import KNNData

from rbm.util.embedding import one_hot_encoding


class MostCommonExperiment:
    def __init__(self, n_labels: int, **kwargs):
        self.n_labels = n_labels
        self.frequency = None

    def fit(self, X):
        self.frequency = one_hot_encoding(X, depth=117, reshape=False).sum(axis=0) / X.shape[0]

    def predict(self, X, y_column):
        return self.predict_proba(X, y_column).argmax(axis=1)

    def predict_proba(self, X, y_column):
        proba = np.zeros((X.shape[0], self.n_labels))
        return proba + self.frequency[y_column]

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
