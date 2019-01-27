from sklearn.base import BaseEstimator


class OtherModel(BaseEstimator):

    def __init__(self):
        self._model = None

    def initialize(self):
        pass

    def fit(self, x, y):
        return self._model.fit(x, y)

    def predict(self, x):
        return self._model.predict(x)

    def predict_proba(self, x):
        return None

    def __repr__(self):
        return self.__class__.__name__


class GenericModel(OtherModel):

    def __init__(self, initialize_function, name: str):
        super().__init__()
        self.initialize_function = initialize_function
        self.name = name

    def initialize(self):
        self._model = self.initialize_function()

    def __repr__(self):
        return self.name
