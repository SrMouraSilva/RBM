import pandas as pd
from sklearn.utils import shuffle
from tqdm import tqdm

from experiments.model_evaluate.grid_search_cv_multi_refit import GridSearchCVMultiRefit
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

                mean = self.method_name(definition, data_train, data_test, evaluate)

                # Save
                name = self._extract_name(definition, i_outer)
                pd.concat(model_results).to_csv(path_save / f'{name}.csv')

    def method_name(self, definition, train, test, evaluate):
        # Measure test for each best params for each metric
        for metric, params in evaluate.best_params().items():
            model = definition.model(*params)

            X_train, y_train = definition.split_method(train)
            X_test, y_test = definition.split_method(test)

            model.fit(X_train)
            metric.eval(model, X_test, y_test)

    def _extract_name(self, definition: TestDefinition, i):
        return f'{definition.__str__()}-({i+1} of {self.cv_outer})'
