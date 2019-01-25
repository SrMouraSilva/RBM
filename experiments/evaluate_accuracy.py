from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier

from experiments.model_evaluate.evaluate_method import Accuracy
from experiments.model_evaluate.model_evaluate import ModelEvaluate
from experiments.other_models.knn import KNNModel
from experiments.other_models.mlm import MinimalLearningMachineClassifier as MLMC
from experiments.other_models.mlm.protosel import ActiveSelection
from experiments.other_models.mlmc import MLMCModel
from experiments.other_models.other_model import GenericModel
from experiments.other_models.rbmcfsvm import RBMCFSVMModel
from experiments.other_models.rbmsvm import RBMSVMModel


def results(model, data):
    data_train = data[data.is_test & data.evaluation.str.contains('train')][list(range(0, 6))]
    data_test = data[data.is_test & data.evaluation.str.contains('test')][list(range(0, 6))]

    print('Model:', model)
    print('')
    print(' - Train:', data_train.mean().mean())
    print(' - Test:', data_test.mean().mean())


models = [
    # Seconds
    KNNModel(total_labels=117, k=1),

    # Minuts
    GenericModel(lambda: svm.SVC(gamma='scale'), 'svm_svc_gamascale'),

    # 10 minutes
    RBMSVMModel(use_probabilities_instead_samples=True),
    RBMCFSVMModel(use_probabilities_instead_samples=True),

    # 20 minutes?
    GenericModel(lambda: MLPClassifier(max_iter=1000), 'mlp'),

    # One hour
    MLMCModel(lambda: MLMC(), 'mlmc-random-selection'),
    # Ten minuts
    MLMCModel(lambda: MLMC(selector=ActiveSelection()), 'mlmc-active-selection'),
]

#C = 10**-5 10**5 pular de 10**2 em 10**2
#gamma = 10**-5 10**5 pular de 10**2 em 10**2




metric = Accuracy()
evaluate = ModelEvaluate(metric)

for model in models:
    print()
    print()
    print('Evaluate', model)
    data = evaluate.evaluate(model)
    data.to_csv(f'other_models/results/{model}-{metric.__class__.__name__}.csv', index=False)
    print(data)
    results(model, data)
