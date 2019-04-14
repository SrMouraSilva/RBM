from abc import ABCMeta, abstractmethod
from warnings import warn

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class ScikitLearnClassifierModel(BaseEstimator, ClassifierMixin):

    def predict(self, X: np.ndarray) -> np.ndarray:
        pass

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        pass


def complete_missing_classes(predictions_with_missing_classes, classes, n_expected_classes, value=0):
    labels = np.array(range(n_expected_classes))
    predictions = predictions_with_missing_classes

    for label in labels:
        if label not in classes:
            predictions = np.insert(predictions, label, value, axis=1)

    return predictions


class EvaluateMethod(metaclass=ABCMeta):
    @abstractmethod
    def evaluate(self, model: ScikitLearnClassifierModel, x, y, **kwargs) -> float:
        pass


class ProbabilisticEvaluateMethod(EvaluateMethod):
    def evaluate(self, model: ScikitLearnClassifierModel, x, y, n_labels=1, **kwargs) -> float:

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

