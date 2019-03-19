from pathlib import Path

import pandas as pd
from sklearn.utils import shuffle
from tqdm import tqdm

from experiments.model_evaluate.grid_search_cv_multi_refit import GridSearchCVMultiRefit, BestParamsResult
from experiments.model_evaluate.test_definition import TestDefinition
from rbm.train.kfold_elements import KFoldElements


class ModelEvaluate:
    """
    Uses GridSearchCV
    """

    def __init__(self, metrics, random_state=42, cv_outer=5, cv_inner=2):
        self.random_state = random_state
        self.cv_outer = cv_outer
        self.cv_inner = cv_inner
        self.metrics = metrics

    def run(self, models: [TestDefinition], data, path_save):
        data = shuffle(data, random_state=self.random_state)

        # Evaluate by model
        for definition in tqdm(models):
            kfolds_outer = KFoldElements(data=data, n_splits=self.cv_outer, random_state=self.random_state, shuffle=False)

            # Outer Cross Validation
            for i_outer, data_train, data_test in kfolds_outer.split():

                # Inner Cross Validation
                evaluate = GridSearchCVMultiRefit(definition, self.random_state, number_of_folds=self.cv_inner, metrics=self.metrics)
                evaluate.fit(data_train)

                results_inner_kfolds = evaluate.results
                results_outer_kfold = self.evaluate_outer_kfold(i_outer, definition, data_train, data_test, evaluate.best_params())

                self._save(definition, i_outer, results_inner_kfolds, results_outer_kfold, path_save)

    def evaluate_outer_kfold(self, i_outer, definition, train, test, best_params: [BestParamsResult]) -> pd.DataFrame:
        _, n_columns = train.shape

        results = []

        # Measure test for each best params for each metric
        for best in best_params:
            for column in range(n_columns):
                model = definition.model(**best.params)

                X_train, y_train = definition.split_method(train, column)
                X_test, y_test = definition.split_method(test, column)

                model.fit(X_train, y_train)

                data = definition.__dict__()
                data.update({
                    'i_outer': i_outer,
                    'column': column,
                    'metric': best.metric.name,
                    'best_params': best.params,
                    'value': best.metric.eval(model, X_test, y_test)
                })

                results.append(data)

        return pd.DataFrame(results)

    def _save(self, definition: TestDefinition, i_outer: int, results_inner_kfolds: pd.DataFrame, results_outer_kfold: pd.DataFrame, path_save: Path):
        name = f'{definition.__str__()}-({i_outer+1} of {self.cv_outer})'

        results_inner_kfolds.to_csv(path_save / 'inner' / f'{name}.csv')
        results_outer_kfold.to_csv(path_save / 'outer' / f'{name}.csv')
