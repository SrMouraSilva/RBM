from experiments.other_models.other_model import GenericModel


class MLMCModel(GenericModel):

    def fit(self, x, y):
        self._model.fit(x+1, y+1)

    def predict(self, x):
        return self._model.predict(x+1) - 1

    def __repr__(self):
        return self.name
