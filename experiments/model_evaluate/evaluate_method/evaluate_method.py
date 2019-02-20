from abc import ABCMeta, abstractmethod
from warnings import warn

import numpy as np
import pandas as pd
from sklearn.metrics import label_ranking_average_precision_score

from experiments.model_evaluate.evaluate_method.some_rank_metrics import dcg_score, average_precision_score
from experiments.other_models.other_model import OtherModel
from experiments.other_models.utils import one_hot_encoding, complete_missing_classes


class EvaluateMethod(metaclass=ABCMeta):
    def evaluate(self, model: OtherModel, x, y, **kwargs) -> float:
        pass


class ProbabilisticEvaluateMethod(EvaluateMethod):
    def evaluate(self, model: OtherModel, x, y, n_labels=1, **kwargs) -> float:

        if getattr(model, "predict_proba", None) is None:
            warn("Model doesn't have predict_proba method. Returns 0")
            return 0

        recommendations = model.predict_proba(x)

        if hasattr(model, 'classes_'):
            recommendations = complete_missing_classes(recommendations, classes=model.classes_, n_expected_classes=n_labels)

        return self.evaluate_probabilistic(y, y_predicted=recommendations, n_labels=n_labels, **kwargs)

    @abstractmethod
    def evaluate_probabilistic(self, y, y_predicted, n_labels, **kwargs):
        pass


class Accuracy(EvaluateMethod):
    def evaluate(self, model: OtherModel, x, y, **kwargs):
        y_generated = model.predict(x)
        return self.accuracy(y, y_generated)

    def accuracy(self, y: pd.DataFrame, y_generated):
        count = sum(y.values == y_generated)

        return count / len(y_generated)


class HitRatio(ProbabilisticEvaluateMethod):
    def __init__(self, k):
        self.k = k

    def evaluate_probabilistic(self, y, y_predicted, n_labels, **kwargs):
        total_instances = y.shape[0]

        y = one_hot_encoding(y, depth=n_labels, dtype=np.bool)
        top_k = self._top_k_mask(y_predicted, n_labels)

        total_correct = np.sum(y & top_k)

        return total_correct / total_instances

    def _top_k_mask(self, y_predicted, n_labels):
        top_k_index = np.argpartition(y_predicted, -self.k)[:, -self.k:]

        top_k_one_hot = one_hot_encoding(top_k_index, depth=n_labels) \
            .reshape([-1, self.k, n_labels]) \
            .sum(axis=1, dtype=np.bool)

        return top_k_one_hot


def hit_ratio_score_function(k, n_labels):
    def mrr_score(estimator: OtherModel, X, y):
        return HitRatio(k).evaluate(estimator, X, y, n_labels)

    return mrr_score


class MRR(ProbabilisticEvaluateMethod):
    """
    Mean Reciprocal Rank
    """

    def evaluate_probabilistic(self, y, y_predicted, n_labels, **kwargs):
        return self.mrr(y, recommendations=y_predicted, depth=n_labels)

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


class MDCG(ProbabilisticEvaluateMethod):
    """
    Mean DCG
    """

    def evaluate_probabilistic(self, y, y_predicted, n_labels, **kwargs):
        y = one_hot_encoding(y, n_labels)

        # Optimize https://codereview.stackexchange.com/a/109577/193386
        dcg_element_wise = np.array([dcg_score(a, b, k=n_labels) for a, b in zip(y, y_predicted)])

        return dcg_element_wise.mean()


def mdcg_score_function(n_labels):
    def mdcg_score(estimator: OtherModel, X, y):
        return MDCG().evaluate(estimator, X, y, n_labels)

    return mdcg_score


class MAP(object):
    """
    Mean Average Precision
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html#sklearn.metrics.average_precision_score
    """

    def __init__(self, k):
        self.k = k

    def evaluate_probabilistic(self, y, y_predicted, n_labels, **kwargs):
        y_categories = self._category_mask_encoding(y, n_labels)

        dcg_element_wise = np.array([average_precision_score(y_categories, y_predicted, k=self.k) for a, b in zip(y, y_predicted)])

        return dcg_element_wise.mean()

    def _category_mask_encoding(self, y, n_labels):
        return one_hot_encoding(y, depth=n_labels)
