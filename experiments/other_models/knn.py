from sklearn.neighbors import KNeighborsClassifier

from experiments.other_models.other_model import OtherModel


class KNNModel(OtherModel):

    def __init__(self, k=1):
        super().__init__()
        self.k = k

    def initialize(self):
        self._model = KNeighborsClassifier(n_neighbors=self.k, algorithm='brute', metric='hamming')
