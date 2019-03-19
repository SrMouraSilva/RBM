import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.utils import shuffle
from tqdm import tqdm

from rbm.train.kfold_elements import KFoldElements


class TestDefinition:
    """
    A test is defined by a model, the parameters that will be applied and the split method
    """
    def __init__(self, model, params, split_method):
        self.model = model
        self.params = params
        self.split_method = split_method

    def __str__(self):
        return f'{self.model.__name__}-{self.split_method.__name__}-{self.params}'


class ModelEvaluate:
    """
    Uses GridSearchCV
    """

    def __init__(self, metrics, random_state=42, cv_outer=5, cv_inner=2):
        self.random_state = random_state
        self.cv_outer = cv_outer
        self.cv_inner = cv_inner
        self.metrics = metrics
        self.n_jobs = n_jobs

    def run(self, models: [TestDefinition], data, path_save):
        data = shuffle(data, random_state=self.random_state)

        # Evaluate by model
        for definition, _model, _params, _split_method in tqdm(models):
            kfolds_outer = KFoldElements(data=data, n_splits=self.cv_outer, random_state=self.random_state, shuffle=False)

            # Outer Cross Validation
            for i_outer, data_train, data_test in kfolds_outer.split():

                # Inner Cross Validation
                evaluate = GridSearchCVMultiRefit(self.random_state, total_of_folds=self.cv_inner, metrics=self.metrics)
                evaluate.fit(definition, data_train)

                # Measure test for each best params for each metric
                for metric, params in evaluate.best_params():
                    model = definition.model(params)
                    #model.fit(data_train)

                    AVALIAR MÉTRICA AQUI

                # Save
                name = self._extract_name(definition, i_outer)
                pd.concat(model_results).to_csv(path_save / f'{name}.csv')

    def _extract_name(self, definition: TestDefinition, i):
        return f'{definition.__str__()}-({i+1} of {self.cv_outer})'


class GridSearchCVMultiRefit:
    """
    Similar a GridSearchCV but detect the best parameters for each metric (scoring) defined

    The MultiRefit name is because the ability of detect the best parameters for more than one parameter
    """

    def __init__(self, random_state, total_of_folds, metrics, n_jobs=-1):
        self.random_state = random_state
        self.total_of_folds = total_of_folds
        self.metrics = metrics
        self.n_jobs = n_jobs

        self.data = []

    def fit(self, definition: TestDefinition, data):
        _, n_columns = data.shape

        # Inner Cross Validation
        # Evaluate by column
        for column in tqdm(range(n_columns)):
            np.random.seed(seed=self.random_state)

            X, y = definition.split_method(data, column)

            clf = self._generate_grid_seach(definition)
            clf.fit(X, y)

            result = self._to_frame(column, clf.cv_results_)
            self.data.append(result)

    def _generate_grid_seach(self, definition: TestDefinition):
        return GridSearchCV(
            definition.model(),
            definition.params,
            cv=self.total_of_folds, scoring=self.metrics,
            n_jobs=self.n_jobs,
            refit=False,
            return_train_score=True
        )

    def _to_frame(self, column, cv_results):
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
        pass

