import numpy as np
from sklearn.linear_model import LogisticRegression

from experiments.model_evaluate.evaluate_method.evaluate_method import complete_missing_classes
from experiments.model_evaluate.evaluate_method.evaluate_method_function import HitRatio
from experiments.multiple_positions_experiment.data import Data

from rbm.util.embedding import one_hot_encoding


class LogisticRegressionData(Data):

    def hide_movies(self, movies: [int]) -> np.ndarray:
        """
        Transform column to missing data
        """
        X = self.data.copy()

        columns = list(range(X.shape[1]))
        for movie in movies:
            columns.remove(movie)

        return X[:, columns]

    def extract_movie(self, movie: int) -> np.ndarray:
        return self.data[:, movie]


class LogisticRegressionExperiment:
    def __init__(self, n_labels: int):
        self.n_labels = n_labels

    def hit_ratio(self, X_train: np.ndarray, X_test: np.ndarray, y_columns: [int], k: int):
        data_train = LogisticRegressionData(X_train, self.n_labels)
        X_train, ys_train = data_train.to_missing_movies(y_columns)
        X_train = one_hot_encoding(X_train, depth=self.n_labels)

        data_test = LogisticRegressionData(X_test, self.n_labels)
        X_test, ys_test = data_test.to_missing_movies(y_columns)
        X_test = one_hot_encoding(X_test, depth=self.n_labels)

        result = {}

        for y_train, y_test, y_column in zip(ys_train, ys_test, y_columns):
            model = LogisticRegression(multi_class='auto', solver='liblinear')
            model.fit(X_train, y_train)

            y_pred = model.predict_proba(X_test)
            y_pred = complete_missing_classes(y_pred, classes=model.classes_, n_expected_classes=self.n_labels)

            y_test = one_hot_encoding(y_test, depth=self.n_labels, dtype=np.bool)
            result[y_column] = HitRatio.hit_ratio(k, y_test, y_pred, self.n_labels)

        return result
