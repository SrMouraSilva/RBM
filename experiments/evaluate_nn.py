from itertools import combinations

import pandas as pd
from sklearn.utils import shuffle
from tqdm import tqdm

from experiments.data.load_data_util import load_data, load_data_categories
from experiments.model_evaluate.evaluate_method.evaluate_method_function import plugins_categories_as_one_hot_encoding
from experiments.nn_experiment.most_common_experiment import MostCommonExperiment
from experiments.nn_experiment.nearest_neighbors_experiment import NearestNeighborsExperiment
from rbm.train.kfold_cross_validation import KFoldCrossValidation

rating_size = 117
total_movies = 6

metrics = []
data = load_data()
plugins_categories = plugins_categories_as_one_hot_encoding(load_data_categories(), rating_size)

data = shuffle(data, random_state=42)
kfolds = KFoldCrossValidation(data=data, n_splits=5, random_state=42, shuffle=False)
params = [
    {'algorithm': 'brute', 'metric': 'hamming', 'n_neighbors': 15},
    {'algorithm': 'brute', 'metric': 'hamming', 'n_neighbors': 25},
    {'algorithm': 'brute', 'metric': 'hamming', 'n_neighbors': 15},
    {'algorithm': 'brute', 'metric': 'hamming', 'n_neighbors': 15},
    {'algorithm': 'brute', 'metric': 'hamming', 'n_neighbors': 15},
]


def metric(fold, metric: str, values: dict):
    results = []

    keys = values.keys()

    for k, v in values.items():
        results += [{
            'fold': fold,
            'y': k,
            'missing': keys - {k},
            'metric': metric,
            'value': v
        }]

    return results


for i, X_train, X_test in kfolds.split():
    #if i != 1:
    #    continue

    model = NearestNeighborsExperiment(rating_size, **params[i])
    #model = MostCommonExperiment(rating_size)
    model.fit(X_train.values)

    for j in tqdm(range(1, 6)):
        for y_columns in combinations(range(6), j):
            metrics += metric(i, 'Accuracy', model.accuracy(X_test.values, y_columns=y_columns))
            metrics += metric(i, 'Hit@5',    model.hit_ratio(X_test.values, y_columns=y_columns, k=5, n_labels=rating_size))
            metrics += metric(i, 'MDCG',     model.mdcg(X_test.values, y_columns=y_columns, n_labels=rating_size))
            metrics += metric(i, 'MAP@5',    model.map(X_test.values, y_columns=y_columns, k=5, n_labels=rating_size, plugins_categories_as_one_hot_encoding=plugins_categories))
            ##metrics['MRR'].append(session.run(model.mrr(X_test.values, y_column=j)))


frame = pd.DataFrame(metrics)
frame.to_csv('nn_results.csv')
#frame.to_csv('common_results.csv')

print(frame.head(5))
#for k, v in metrics.items():
#    print(k.ljust(10), np.mean(v), np.std(v))
