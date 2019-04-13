from typing import Iterator

from sklearn.model_selection import KFold


class KFoldCrossValidation:
    """
    Split the data in k-folds.
    While the original sklearn.model_selection.KFold returns the index in the iteration,
    an instance of this class returns the objects instead the indexes.
    """

    def __init__(self, data, n_splits: int, random_state: int, shuffle: bool):
        self._iterator = None

        if n_splits == 1:
            self.has_splits = False
        else:
            self.has_splits = True
            self.kfold = KFold(n_splits=n_splits, random_state=random_state, shuffle=shuffle)

        self.data = data

    def split(self):
        if self.has_splits:
            self._iterator = enumerate(self.kfold.split(self.data))
        else:
            self._iterator = enumerate([self.data])

        return self

    def __iter__(self) -> Iterator:
        return self

    def __next__(self):
        if self.has_splits:
            index, (train_index, test_index) = self._iterator.__next__()
            training = self.data.iloc[train_index]
            test = self.data.iloc[test_index]

            return index, training, test
        else:
            index, data = self._iterator.__next__()
            return index, data, None
