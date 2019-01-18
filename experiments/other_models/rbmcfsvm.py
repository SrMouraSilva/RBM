from experiments.other_models.rbmsvm import RBMSVMModel
from rbm.rbmcf import RBMCF
from rbm.sampling.contrastive_divergence import ContrastiveDivergence


class RBMCFSVMModel(RBMSVMModel):

    def __init__(self, samples=False):
        super().__init__(RBMCF, samples=samples)
        self._models_path = [
            f'../results/model/kfold={i}+kfold-intern=0+batch_size=10+class={RBMCF.__name__}+visible_size=702+hidden_size=1000+regularization=NoRegularization-0.0+learning_rate=ConstantLearningRate-0.2+sampling_method=CD-1+momentum=1/rbm.ckpt'
            for i in range(0, 5) for _ in range(0, 6)
        ]

    def _initialize_rbm(self, model_path: str):
        self._rbm = RBMCF(
            movies_size=6,
            ratings_size=int(702 / 6),
            hidden_size=1000,
            regularization=None,
            learning_rate=0.2,
            sampling_method=ContrastiveDivergence(1),
            momentum=1
        )
        self._rbm.load(self._session, model_path)
