from abc import ABCMeta

import pandas as pd
from sklearn.metrics import label_ranking_average_precision_score

from experiments.other_models.other_model import OtherModel
from experiments.other_models.utils import one_hot_encoding, complete_missing_classes


class EvaluateMethod(metaclass=ABCMeta):
    def evaluate(self, model: OtherModel, x, y, label=None) -> float:
        pass


class Accuracy(EvaluateMethod):
    def evaluate(self, model: OtherModel, x, y, label=None):
        y_generated = model.predict(x)
        return self.accuracy(y, y_generated)

    def accuracy(self, y: pd.DataFrame, y_generated):
        count = sum(y.values == y_generated)

        return count / len(y_generated)


class MRR(EvaluateMethod):
    """
    Mean Reciprocal Rank
    """

    def evaluate(self, model: OtherModel, x, y, n_labels=0):
        recommendations = model.predict_proba(x)

        if hasattr(model, 'classes_'):
            recommendations = complete_missing_classes(recommendations, classes=model.classes_, n_expected_classes=n_labels)

        return self.mrr(y, recommendations, n_labels)

    def mrr(self, y, recommendations, depth):
        y = one_hot_encoding(y, depth)
        # If there is exactly one relevant label per sample, label ranking average precision is
        # equivalent to the mean reciprocal rank
        # https://scikit-learn.org/stable/modules/model_evaluation.html#label-ranking-average-precision
        score = label_ranking_average_precision_score(y, recommendations)

        return score


def mrr_score_function(n_labels):
    def mrr_score(estimator: OtherModel, X, y):
        return MRR().evaluate(estimator, X, y, n_labels)

    return mrr_score
