from sklearn import svm
from sklearn.neural_network import MLPClassifier

from experiments.other_models.knn import KNNModel
from experiments.other_models.mlm.selectors import KSSelection, NLSelection
from experiments.other_models.model_evaluate import ModelEvaluate
from experiments.other_models.other_model import GenericModel

from mlm import MinimalLearningMachineClassifier as MLMC


def results(model, data):
    data_train = data[data.is_test & data.evaluation.str.contains('train')][list(range(0, 6))]
    data_test = data[data.is_test & data.evaluation.str.contains('test')][list(range(0, 6))]

    print('Model:', model)
    print('')
    print(' - Train:', data_train.mean().mean())
    print(' - Test:', data_test.mean().mean())


models = [
    KNNModel(),
    #GenericModel(lambda: MLMC(), 'mlmc'),
    GenericModel(lambda: MLMC(selector=KSSelection()), 'mlmc-KSSelection'),
    GenericModel(lambda: MLMC(selector=NLSelection()), 'mlmc-NLSelection'),
    ##GenericModel(lambda: svm.LinearSVC(), 'svm_linear'),
    #GenericModel(lambda: svm.SVC(gamma='scale'), 'svm_svc_gamascale'),
    ##GenericModel(lambda: svm.NuSVC(), 'svm_nusvc_gamascale'),
    #GenericModel(lambda: MLPClassifier(max_iter=500), 'mlp'),
]

evaluate = ModelEvaluate()

for model in models:
    print()
    print()
    print('Evaluate', model)
    data = evaluate.evaluate(model)
    print(data)

    results(model, data)
