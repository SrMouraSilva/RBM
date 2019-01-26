import numpy as np
from sklearn import svm
from sklearn.random_projection import GaussianRandomProjection

from experiments.other_models.other_model import OtherModel
from experiments.other_models.utils import x_as_one_hot_encoding


class SVMModel(OtherModel):

    def __init__(self, **params):
        super().__init__()
        self._params = params

    def initialize(self):
        self._model = svm.SVC(**self._params)

    def __repr__(self):
        return super().__repr__() + str(self._params)


class SVMRandomMatrix(SVMModel):

    def __init__(self, **params):
        super().__init__(**params)
        COLUMNS = 6
        self._random_matrix = np.random.rand(COLUMNS - 1, COLUMNS - 1)

    def initialize(self):
        self._model = svm.SVC(**self._params)

    def fit(self, x, y):
        return super().fit(x @ self._random_matrix, y)

    def predict(self, x):
        return self._model.predict(x @ self._random_matrix)


class SVMBagOfWordsGaussianRandom(SVMModel):
    def __init__(self, **params):
        super().__init__(**params)
        self._transformer = GaussianRandomProjection(n_components=50)

    def fit(self, x, y):
        x = x_as_one_hot_encoding(x, categories=117)
        x = self._transformer.fit_transform(x)
        return super().fit(x, y)

    def predict(self, x):
        x = x_as_one_hot_encoding(x, categories=117)
        x = self._transformer.fit_transform(x)

        return self._model.predict(x)
