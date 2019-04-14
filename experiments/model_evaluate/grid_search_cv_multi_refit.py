import ast
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm

from experiments.model_evaluate.evaluate_method.evaluate_method import ScikitLearnClassifierModel
from experiments.model_evaluate.test_definition import TestDefinition


class Metric:
    def __init__(self, name: str, method: Callable[[ScikitLearnClassifierModel, pd.DataFrame, pd.Series], float]):
        self.name = name
        self.method = method

    def eval(self, model: ScikitLearnClassifierModel, X: pd.DataFrame, y: pd.Series):
        return self.method(model, X, y)


class BestParamsResult:

    def __init__(self, definition: TestDefinition, params: dict, metric: Metric):
        self.model = definition.model
        self.params = params
        self.metric = metric

        self.possible_params = definition.params


class GridSearchCVMultiRefit:
    """
    Similar a GridSearchCV but detect the best parameters for each metric (scoring) defined

    The MultiRefit name is because the ability of detect the best parameters for more than one parameter
    """

    def __init__(self, definition: TestDefinition, random_state: int, number_of_folds: int, metrics: dict, n_jobs=-1):
        self.definition = definition

        self.random_state = random_state
        self.number_of_folds = number_of_folds
        self._metrics = metrics
        self.n_jobs = n_jobs

        self.results: pd.DataFrame = None

    def fit(self, data):
        _, n_columns = data.shape
        results = []

        # Inner Cross Validation
        # Evaluate by column
        for column in tqdm(range(n_columns)):
            np.random.seed(seed=self.random_state)

            X, y = self.definition.split_method(data, column)

            clf = self._generate_grid_seach(self.definition)
            clf.fit(X, y)

            result = self._to_frame(column, clf.cv_results_)
            results.append(result)

        self.results = pd.concat(results)

    def _generate_grid_seach(self, definition: TestDefinition):
        return GridSearchCV(
            definition.model(),
            definition.params,
            cv=self.number_of_folds, scoring=self._metrics,
            n_jobs=self.n_jobs,
            refit=False,
            return_train_score=True
        )

    def _to_frame(self, column: int, cv_results: dict):
        data = {
            'column': column,
        }

        data.update(cv_results)

        return pd.DataFrame(data)

    def best_params(self) -> [BestParamsResult]:
        """
        Best params for each metric
        A parameter is selected as best by the highest mean of columns evaluates
        """
        best_params = []

        for metric in self.metrics:
            metric_column = f'mean_test_{metric.name}'
            self.results['params_str'] = self.results['params'].map(str)

            best_param_string = self.results.groupby('params_str')[metric_column]\
                .mean()\
                .idxmax()

            params = ast.literal_eval(best_param_string)
            best = BestParamsResult(self.definition, params, metric)

            best_params.append(best)

        return best_params

    @property
    def metrics(self):
        return [Metric(metric_name, metric_function) for metric_name, metric_function in self._metrics.items()]
