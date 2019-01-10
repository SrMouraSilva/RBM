from sklearn.neighbors import KNeighborsClassifier

from experiments.other_models.other_model import OtherModel


class KNNModel(OtherModel):

    def __init__(self, k=1):
        self.k = k
        self.nbrs: KNeighborsClassifier = None

    def reset(self):
        self.nbrs = KNeighborsClassifier(n_neighbors=self.k, algorithm='brute', metric='hamming')

    def fit(self, x, y):
        self.nbrs.fit(x, y)

    def predict(self, x):
        return self.nbrs.predict(x)
