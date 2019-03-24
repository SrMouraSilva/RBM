from abc import ABCMeta, abstractmethod
from warnings import warn

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import label_ranking_average_precision_score

from experiments.model_evaluate.evaluate_method.some_rank_metrics import dcg_score, average_precision_score
from experiments.other_models.other_model import OtherModel
from experiments.other_models.utils import one_hot_encoding, complete_missing_classes, k_hot_encoding


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


def accuracy(estimator: OtherModel, X, y):
    y_pred = estimator.predict(X)
    return accuracy_score(y, y_pred, normalize=True)


class HitRatio(ProbabilisticEvaluateMethod):
    def __init__(self, k):
        self.k = k

    def evaluate_probabilistic(self, y, y_predicted, n_labels, **kwargs):
        y = one_hot_encoding(y, depth=n_labels, dtype=np.bool)
        return HitRatio.hit_ratio(self.k, y, y_predicted, n_labels=n_labels)

    @staticmethod
    def hit_ratio(k: int, y: np.ndarray, y_predicted: np.ndarray, n_labels: int):
        total_instances = y.shape[0]

        k_hot = k_hot_encoding(k, y_predicted, n_labels)

        total_correct = np.sum(y & k_hot)

        return total_correct / total_instances


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


class MAP(ProbabilisticEvaluateMethod):
    """
    Mean Average Precision
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html#sklearn.metrics.average_precision_score
    """

    def __init__(self, k, class_categories_as_one_hot):
        self.k = k
        self.class_categories_as_one_hot = class_categories_as_one_hot

    def evaluate_probabilistic(self, y, y_predicted, n_labels, **kwargs):
        y_categories = self._category_mask_encoding(y, n_labels)
        y_predicted = self._recomended_mask_encoding(y_predicted, y_categories, n_labels)

        dcg_element_wise = np.array([average_precision_score(a, b, k=self.k) for a, b in zip(y_categories.values, y_predicted.values)])

        return dcg_element_wise.mean()

    def _recomended_mask_encoding(self, y_predicted, y_categories, n_labels):
        """
        Activate the k-most probable items that with the same category of expected
        """
        # k most probable items
        top_k = np.flip(np.argsort(y_predicted), axis=1)[:, :self.k]
        # activate the top_k based in y category
        top_k_as_one_hot = one_hot_encoding(top_k, n_labels, reshape=False).sum(axis=1).astype(np.bool)
        # All in top-k with the same category of y
        return y_categories & top_k_as_one_hot

    def _category_mask_encoding(self, y, n_labels):
        columns = range(n_labels)
        return y.to_frame().join(self.class_categories_as_one_hot, on=y.name)[columns]


def map_score_function(k: int, n_labels: int, categories: pd.DataFrame):
    columns = range(n_labels)

    eye = pd.DataFrame(np.eye(n_labels), columns=columns)
    categories_as_one_hot = categories.join(eye).groupby('category').sum().astype(np.int32)
    plugins_categories_as_one_hot = categories.join(categories_as_one_hot, on='category')[columns]

    def map_score(estimator: OtherModel, X, y):
        return MAP(k, plugins_categories_as_one_hot).evaluate(estimator, X, y, n_labels)

    return map_score
