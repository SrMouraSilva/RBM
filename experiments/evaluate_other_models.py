from sklearn import svm
from sklearn.neural_network import MLPClassifier

from experiments.other_models.extractor import ExtractorModel
from experiments.other_models.knn import KNNModel
from experiments.other_models.mlm.protosel import KSSelection, NLSelection, RegEnnSelection, ActiveSelection, DROP2_RE, MutualInformationSelection
from experiments.other_models.mlmc import MLMCModel
from experiments.other_models.model_evaluate import ModelEvaluate
from experiments.other_models.other_model import GenericModel

from experiments.other_models.mlm import MinimalLearningMachineClassifier as MLMC
from experiments.other_models.mlm import NearestNeighborMinimalLearningMachineClassifier as NNMLMC
from experiments.other_models.rbmcfsvm import RBMCFSVMModel
from experiments.other_models.rbmsvm import RBMSVMModel
from rbm.rbm import RBM
from rbm.rbmcf import RBMCF


def results(model, data):
    data_train = data[data.is_test & data.evaluation.str.contains('train')][list(range(0, 6))]
    data_test = data[data.is_test & data.evaluation.str.contains('test')][list(range(0, 6))]

    print('Model:', model)
    print('')
    print(' - Train:', data_train.mean().mean())
    print(' - Test:', data_test.mean().mean())


models = [
    # Seconds
    KNNModel(),

    ##GenericModel(lambda: svm.LinearSVC(), 'svm_linear'),
    # Minuts
    GenericModel(lambda: svm.SVC(gamma='scale'), 'svm_svc_gamascale'),
    ##GenericModel(lambda: svm.NuSVC(), 'svm_nusvc_gamascale'),

    # 10 minutes
    RBMSVMModel(),
    RBMCFSVMModel(),

    RBMSVMModel(samples=False),
    RBMCFSVMModel(samples=False),

    # 20 minutes?
    GenericModel(lambda: MLPClassifier(max_iter=500), 'mlp'),

    ##GenericModel(lambda: NNMLMC(), 'nnmlmc'),
    # One hour
    MLMCModel(lambda: MLMC(), 'mlmc-random-selection'),
    # Ten minuts
    MLMCModel(lambda: MLMC(selector=ActiveSelection()), 'mlmc-active-selection'),
    #MLMC - possible selectos (KSSelection, NLSelection, RegEnnSelection, ActiveSelection, DROP2_RE, MutualInformationSelection)
]
#models = [ExtractorModel()]
models = [RBMCFSVMModel(samples=False)]

evaluate = ModelEvaluate()

for model in models:
    print()
    print()
    print('Evaluate', model)
    data = evaluate.evaluate(model)
    print(data)

    results(model, data)
