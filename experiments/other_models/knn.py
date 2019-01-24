import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from experiments.other_models.other_model import OtherModel


class KNNModel(OtherModel):

    def __init__(self, total_labels, k=1):
        super().__init__()
        self.total_labels = total_labels
        self.k = k

    def initialize(self):
        self._model = KNeighborsClassifier(n_neighbors=self.k, algorithm='brute', metric='hamming')

    def recommends(self, x):
        labels = np.array(range(self.total_labels))

        predictions_with_missing_classes = self._model.predict_proba(x)
        predictions = predictions_with_missing_classes

        for label in labels:
            if label not in self._model.classes_:
                predictions = np.insert(predictions, label, 0, axis=1)

        return predictions
