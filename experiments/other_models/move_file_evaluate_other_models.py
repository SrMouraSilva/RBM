from experiments.other_models.knn import KNNModel
from experiments.other_models.model_evaluate import ModelEvaluate

knn = KNNModel()

evaluate = ModelEvaluate()

evaluate.evaluate(knn)
