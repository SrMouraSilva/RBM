import tensorflow as tf
from sklearn import svm

from experiments.other_models.rbm_model import RBMOtherModel
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
            learning_rate=0.1,
            sampling_method=ContrastiveDivergence(1),
            momentum=1
        )

    def fit(self, x, y):
        column = self._current_train % 6

        x2 = x.copy()
        x2.insert(column, 'y', y)

        x2 = tf.one_hot(x2, depth=117).reshape((-1, 117*6)).eval(session=self._session)

        if self.use_probabilities_instead_samples:
            hidden = self._rbm.P_h_given_v(x2.T).eval(session=self._session)
        else:
            hidden = self._rbm.sample_h_given_v(x2.T).eval(session=self._session)

        self._model.fit(hidden.T, y)

    def predict(self, x):
        x = self.recommends(x)

        if self.use_probabilities_instead_samples:
            hidden = self._rbm.P_h_given_v(x.T).eval(session=self._session)
        else:
            hidden = self._rbm.sample_h_given_v(x.T).eval(session=self._session)

        return self._model.predict(hidden.T)
