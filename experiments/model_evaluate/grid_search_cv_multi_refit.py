import ast

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm

from experiments.model_evaluate.test_definition import TestDefinition


class GridSearchCVMultiRefit:
    """
    Similar a GridSearchCV but detect the best parameters for each metric (scoring) defined

    The MultiRefit name is because the ability of detect the best parameters for more than one parameter
    """

    def __init__(self, random_state: int, number_of_folds: int, metrics: dict, n_jobs=-1):
        self.random_state = random_state
        self.number_of_folds = number_of_folds
        self.metrics = metrics
        self.n_jobs = n_jobs

        self.results: pd.DataFrame = None

    def fit(self, definition: TestDefinition, data):
        _, n_columns = data.shape
        results = []

        # Inner Cross Validation
        # Evaluate by column
        for column in tqdm(range(n_columns)):
            np.random.seed(seed=self.random_state)

            X, y = definition.split_method(data, column)

            clf = self._generate_grid_seach(definition)
            clf.fit(X, y)

            result = self._to_frame(column, clf.cv_results_)
            results.append(result)

        self.results = pd.concat(results)

    def _generate_grid_seach(self, definition: TestDefinition):
        return GridSearchCV(
            definition.model(),
            definition.params,
            cv=self.number_of_folds, scoring=self.metrics,
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

    def best_params(self):
        """
        Best params for each metric
        A parameter is selected as best by the highest mean of columns evaluates
        """
        best_params = dict()

        for metric in self.metrics.keys():
            metric_column = f'mean_test_{metric}'
            self.results['params_str'] = self.results['params'].map(str)

            best_param_string = self.results.groupby('params_str')[metric_column]\
                .mean()\
                .idxmax()

            best_params[metric] = ast.literal_eval(best_param_string)

        return best_params
