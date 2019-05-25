from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss
from sklearn.metrics import label_ranking_average_precision_score

from experiments.model_evaluate.evaluate_method.evaluate_method import ScikitLearnClassifierModel, \
    ProbabilisticEvaluateMethod
from experiments.model_evaluate.evaluate_method.some_rank_metrics import dcg_score, average_precision_score
from rbm.util.embedding import one_hot_encoding, k_hot_encoding


MetricFunction = Callable[[ScikitLearnClassifierModel, pd.DataFrame, pd.Series], float]


def accuracy(estimator: ScikitLearnClassifierModel, X, y):
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


def hit_ratio_score_function(k, n_labels) -> MetricFunction:
    def hit_ratio(estimator: ScikitLearnClassifierModel, X, y):
        return HitRatio(k).evaluate(estimator, X, y, n_labels)

    return hit_ratio


class MRR(ProbabilisticEvaluateMethod):
    """
    Mean Reciprocal Rank
    """

    def evaluate_probabilistic(self, y, y_predicted, n_labels, **kwargs):
        y = one_hot_encoding(y, depth=n_labels)
        return MRR.mean_reciprocal_rank(y, recommendations=y_predicted)

    @staticmethod
    def mean_reciprocal_rank(y, recommendations):
        # If there is exactly one relevant label per sample, label ranking average precision is
        # equivalent to the mean reciprocal rank
        # https://scikit-learn.org/stable/modules/model_evaluation.html#label-ranking-average-precision
        return label_ranking_average_precision_score(y, recommendations)


def mrr_score_function(n_labels) -> MetricFunction:
    def mrr_score(estimator: ScikitLearnClassifierModel, X, y):
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

    @staticmethod
    def mdcg(y, y_predicted, n_labels):
        dcg_element_wise = np.array([dcg_score(a, b, k=n_labels) for a, b in zip(y, y_predicted)])

        return dcg_element_wise.mean()


def mdcg_score_function(n_labels) -> MetricFunction:
    def mdcg_score(estimator: ScikitLearnClassifierModel, X, y):
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
        if self.k == 1:
            top_k_as_one_hot = one_hot_encoding(top_k, n_labels, reshape=False).astype(np.bool)
        else:
            top_k_as_one_hot = one_hot_encoding(top_k, n_labels, reshape=False).sum(axis=1).astype(np.bool)
        # All in top-k with the same category of y
        return y_categories & top_k_as_one_hot

    def _category_mask_encoding(self, y, n_labels):
        columns = range(n_labels)
        return y.to_frame().join(self.class_categories_as_one_hot, on=y.name)[columns]


def map_score_function(k: int, n_labels: int, categories: pd.DataFrame) -> MetricFunction:
    plugins_categories_as_one_hot = plugins_categories_as_one_hot_encoding(categories, n_labels)

    def map_score(estimator: ScikitLearnClassifierModel, X, y):
        return MAP(k, plugins_categories_as_one_hot).evaluate(estimator, X, y, n_labels)

    return map_score


def plugins_categories_as_one_hot_encoding(categories: pd.DataFrame, n_labels: int):
    columns = range(n_labels)

    eye = pd.DataFrame(np.eye(n_labels), columns=columns)
    categories_as_one_hot = categories.join(eye).groupby('category').sum().astype(np.int32)
    return categories.join(categories_as_one_hot, on='category')[columns]


def cross_entropy_function() -> MetricFunction:
    def cross_entropy(estimator: ScikitLearnClassifierModel, X, y):
        y_predict = estimator.predict_proba(X)

        return log_loss(y_true=y, y_pred=y_predict, labels=estimator.classes_)

    return cross_entropy


def permutation_accuracy_score(X, permutations, energies, n_labels, non_fixed_column: tuple):
    if X.shape[0] != energies.shape[0]:
        raise Exception(f'Expected number of energies (energies.shape[0]={energies.shape[0]})'
                        f' equals to total of elements (X.shape[0]={X.shape[0]})')

    if permutations.shape[0] != energies.shape[1]:
        raise Exception(f'Expected number of permutations (permutations.shape[0]={permutations.shape[0]})'
                        f' equals to total of energies for element (energies.shape[1]={energies.shape[1]})')

    most_probability = energies.argmin(axis=1)
    selected_permutations = permutations[most_probability]

    X_not_one_hot = X.reshape([X.shape[0], -1, n_labels]).argmax(axis=2)

    X_recommended = np.apply_along_axis(lambda x, i: x[next(i)], 1, X_not_one_hot, selected_permutations.__iter__())

    X_expected_columns = X_not_one_hot[:, non_fixed_column]
    X_recommended_columns = X_recommended[:, non_fixed_column]

    total_of_elements = (X_recommended_columns.shape[0] * X_recommended_columns.shape[1])

    return np.equal(X_expected_columns, X_recommended_columns).sum() / total_of_elements
