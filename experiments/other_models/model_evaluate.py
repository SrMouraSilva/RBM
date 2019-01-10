from typing import Iterator

import pandas as pd

from sklearn.model_selection import KFold
from sklearn.utils import shuffle

from experiments.other_models.other_model import OtherModel


class KFoldElements:

    def __init__(self, data, n_splits: int, random_state: int, shuffle: bool):
        self._iterator = None

        self.kfold = KFold(n_splits=n_splits, random_state=random_state, shuffle=shuffle)
        self.data = data

    def split(self):
        self._iterator = enumerate(self.kfold.split(self.data))
        return self

    def __iter__(self) -> Iterator:
        return self

    def __next__(self):
        index, (train_index, test_index) = self._iterator.__next__()
        training = self.data.iloc[train_index]
        test = self.data.iloc[test_index]

        return index, training, test


class ModelEvaluate:

    def __init__(self, random_state=42, columns=6):
        self.random_state = random_state
        self.columns = columns

    def _read_data(self, path, index_col=None):
        if index_col is None:
            index_col = ['id', 'name']

        return pd.read_csv(path, sep=",", index_col=index_col)

    def evaluate(self, model):
        data = self._read_data('../data/pedalboard-plugin.csv')

        data_shuffled = shuffle(data, random_state=self.random_state)

        kfolds_training_test = KFoldElements(data=data_shuffled, n_splits=5, random_state=self.random_state, shuffle=False)

        for i, original_training, test in kfolds_training_test.split():
            kfolds_training_validation = KFoldElements(data=original_training, n_splits=2, random_state=self.random_state, shuffle=False)

            for j, training, validation in kfolds_training_validation.split():
                print(i, j, self.evaluate_by_column(model, training, validation))

            print(i, self.evaluate_by_column(model, original_training, test))

    def evaluate_by_column(self, model: OtherModel, training, test):
        result = []

        for column in range(self.columns):
            x_train, y_train = self._split_x_y(training, column)
            x_test, y_expected = self._split_x_y(test, column)

            model.reset()
            model.fit(x_train, y_train)
            y_generated = model.predict(x_test)

            total_equals = sum(y_expected.values == y_generated)

            result.append([column, total_equals/len(y_generated)])

        return result

    def _split_x_y(self, data, test_column_index):
        columns = [f'plugin{i}' for i in range(1, self.columns+1)]
        train_columns = columns[1:test_column_index] + columns[test_column_index+1:self.columns+1]
        test_column = f'plugin{test_column_index+1}'

        return data[train_columns], data[test_column]
