import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.utils import shuffle
from tqdm import tqdm


class ModelEvaluate:
    """
    Uses GridSearchCV
    """

    def __init__(self, metrics, random_state=42, cv=5, n_jobs=-1):
        self.random_state = random_state
        self.cv = cv
        self.metrics = metrics
        self.n_jobs = n_jobs

    def run(self, models, data, path_save):
        data = shuffle(data, random_state=self.random_state)
        n_samples, n_columns = data.shape

        # Evaluate by model
        for model, params, split_method in tqdm(models):
            name = self._extract_name(model, split_method, params)
            model_results = []

            # Evaluate by column
            for column in tqdm(range(n_columns)):
                np.random.seed(seed=self.random_state)

                X, y = split_method(data, column)

                clf = GridSearchCV(model(), params, cv=self.cv, n_jobs=self.n_jobs, scoring=self.metrics, refit=False, return_train_score=True)
                clf.fit(X, y)

                result = self._extract_result(name, column, clf.cv_results_)

                model_results.append(result)

            # Save
            pd.concat(model_results).to_csv(path_save / f'{name}.csv')

    def _extract_name(self, model, split_method, params):
        return f'{model.__name__}-{split_method.__name__}-{params}'

    def _extract_result(self, name, column, cv_results):
        data = {
            'column': column,
            'evaluation_method': name
        }

        data.update(cv_results)

        return pd.DataFrame(data)
