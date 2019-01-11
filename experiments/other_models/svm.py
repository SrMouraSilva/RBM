from sklearn import svm

from experiments.other_models.other_model import OtherModel


class SVMModel(OtherModel):

    def initialize(self):
        self._model = svm.SVC(gamma='scale')
