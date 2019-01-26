import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.utils import shuffle
from tqdm import tqdm


class SimpleModelEvaluate:
    """
    Uses GridSearchCV
    """

    def __init__(self, metrics=None, random_state=42, cv=5):
        self.random_state = random_state
        self.cv = cv

        if metrics is None:
            self.metrics = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
        else:
            self.metrics = metrics

    def run(self, models, data, path_save):
        data = shuffle(data, random_state=self.random_state)
        n_samples, n_columns = data.shape

        for model, params, split_method in tqdm(models):
            name = ""
            model_results = []

            for column in tqdm(range(n_columns)):
                np.random.seed(seed=self.random_state)

                X, y = split_method(data, column)

                clf = GridSearchCV(model(), params, cv=self.cv, n_jobs=-1, scoring=self.metrics, refit=self.metrics[0], return_train_score=True)
                clf.fit(X, y)

                name = self._extract_name(model, split_method, params)
                result = self.extract_result(name, column, clf.cv_results_)

                model_results.append(result)

            pd.concat(model_results).to_csv(path_save / f'{name}.csv')


    def _extract_name(self, model, split_method, params):
        return f'{model.__name__}-{split_method.__name__}-{params}'

    def extract_result(self, name, column, cv_results):
        data = {
            'column': column,
            'is_test': False,
            'evaluation': 'train',
            'evaluation_method': name
        }

        data.update(cv_results)

        return pd.DataFrame(data)

