import tensorflow as tf
from sklearn import svm

from experiments.other_models.other_model import OtherModel
from rbm.rbm import RBM
from rbm.sampling.contrastive_divergence import ContrastiveDivergence


class RBMSVMModel(OtherModel):

    def __init__(self, rbm_class=RBM, samples=True):
        super().__init__()
        self._models_path = [
            f'../results/model/kfold={i}+kfold-intern=0+batch_size=10+class={rbm_class.__name__}+visible_size=702+hidden_size=1000+regularization=NoRegularization-0.0+learning_rate=ConstantLearningRate-0.1+sampling_method=CD-1+momentum=1/rbm.ckpt'
            for i in range(0, 5) for _ in range(0, 6)
        ]
        self._rbm: RBM = None
        self._session = None
        self._current_train = -1

        self.samples = samples

    def initialize(self):
        self._current_train += 1

        model_path = self._models_path[0]
        del self._models_path[0]

        if self._rbm is not None:
            self._session.close()
            tf.reset_default_graph()

        self._session = tf.Session()

        self._initialize_rbm(model_path)
        self._model = svm.SVC(gamma='scale')

    def _initialize_rbm(self, model_path: str):
        self._rbm = RBM(
            visible_size=702,
            hidden_size=1000,
            regularization=None,
            learning_rate=0.1,
            sampling_method=ContrastiveDivergence(1),
            momentum=1
        )
        self._rbm.load(self._session, model_path)

    def fit(self, x, y):
        column = self._current_train % 6

        x2 = x.copy()
        x2.insert(column, 'y', y)

        x2 = tf.one_hot(x2, depth=117).reshape((-1, 117*6)).eval(session=self._session)

        if self.samples:
            hidden = self._rbm.sample_h_given_v(x2.T).eval(session=self._session)
        else:
            hidden = self._rbm.P_h_given_v(x2.T).eval(session=self._session)

        self._model.fit(hidden.T, y)

    def predict(self, x):
        column = self._current_train % 6

        x2 = x.copy()

        x2.insert(column, 'y', [0]*x.shape[0])
        x2 = tf.one_hot(x2, depth=117).reshape((-1, 117 * 6)).eval(session=self._session)

        if self.samples:
            hidden = self._rbm.sample_h_given_v(x2.T).eval(session=self._session)
        else:
            hidden = self._rbm.P_h_given_v(x2.T).eval(session=self._session)

        return self._model.predict(hidden.T)
