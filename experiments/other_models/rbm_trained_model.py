from abc import ABCMeta

import numpy as np

from experiments.other_models.rbm_model import RBMOtherModel
from rbm.learning.constant_learning_rate import ConstantLearningRate
from rbm.rbm import RBM
from rbm.rbmcf import RBMCF
from rbm.sampling.contrastive_divergence import ContrastiveDivergence


class RBMAlreadyTrainedModelBase(RBMOtherModel, metaclass=ABCMeta):

    def predict(self, x):
        v0 = self.prepare_x_as_one_hot_encoding(x.copy(), column=self.column).T

        p_h = self._rbm.P_h_given_v(v0)
        p_v1 = self._rbm.P_v_given_h(p_h)

        y_generated = p_v1[self.column * 117:(self.column + 1) * 117]
        y_generated = y_generated.T.eval(session=self._session)

        v1_as_label = np.argmax(y_generated, axis=1)
        return v1_as_label


class RBMAlreadyTrainedModel(RBMAlreadyTrainedModelBase):

    def __init__(self):
        super().__init__(self._create_function())

    def _create_function(self):
        return lambda: RBM(
            visible_size=117 * 6,
            hidden_size=1000,
            regularization=None,
            learning_rate=ConstantLearningRate(0.1),
            sampling_method=ContrastiveDivergence(1),
            momentum=1
        )


class RBMCFAlreadyTrainedModel(RBMAlreadyTrainedModelBase):

    def __init__(self):
        super().__init__(self._create_function())

    def _create_function(self):
        return lambda: RBMCF(
            movies_size=6,
            ratings_size=int(702 / 6),
            hidden_size=1000,
            regularization=None,
            learning_rate=ConstantLearningRate(0.2),
            sampling_method=ContrastiveDivergence(1),
            momentum=1
        )
