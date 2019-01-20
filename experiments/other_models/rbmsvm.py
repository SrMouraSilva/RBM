from sklearn import svm

from experiments.other_models.rbm_model import RBMOtherModel
from rbm.learning.constant_learning_rate import ConstantLearningRate
from rbm.rbm import RBM
from rbm.sampling.contrastive_divergence import ContrastiveDivergence


class RBMSVMModel(RBMOtherModel):

    def __init__(self, use_probabilities_instead_samples=False):
        super().__init__(self._create_function())
        self.use_probabilities_instead_samples = use_probabilities_instead_samples

    def initialize(self):
        super().initialize()
        self._model = svm.SVC(gamma='scale')

    def _create_function(self):
        return lambda: RBM(
            visible_size=117*6,
            hidden_size=1000,
            regularization=None,
            learning_rate=ConstantLearningRate(0.1),
            sampling_method=ContrastiveDivergence(1),
            momentum=1
        )

    def fit(self, x, y):
        x = self.prepare_x_as_one_hot_encoding(x.copy(), column=self.column, column_data=y)
        hidden = self.hidden_from(x)

        self._model.fit(hidden.T, y)

    def predict(self, x):
        x = self.prepare_x_as_one_hot_encoding(x.copy(), column=self.column)
        hidden = self.hidden_from(x)

        return self._model.predict(hidden.T)

    def hidden_from(self, x):
        if self.use_probabilities_instead_samples:
            return self._rbm.P_h_given_v(x.T).eval(session=self._session)
        else:
            return self._rbm.sample_h_given_v(x.T).eval(session=self._session)
