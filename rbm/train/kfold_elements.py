from typing import Iterator

from sklearn.model_selection import KFold


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
