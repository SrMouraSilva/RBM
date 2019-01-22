from experiments.other_models.rbmsvm import RBMSVMModel
from rbm.learning.constant_learning_rate import ConstantLearningRate
from rbm.rbmcf import RBMCF
from rbm.sampling.contrastive_divergence import ContrastiveDivergence


class RBMCFSVMModel(RBMSVMModel):

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
