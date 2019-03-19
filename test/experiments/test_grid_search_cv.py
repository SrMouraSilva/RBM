import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

from experiments.model_evaluate.evaluate_method.evaluate_method import mdcg_score_function, mrr_score_function, \
    hit_ratio_score_function
from experiments.model_evaluate.grid_search_cv_multi_refit import GridSearchCVMultiRefit
from experiments.model_evaluate.split_method import split_x_y
from experiments.model_evaluate.test_definition import TestDefinition

data = pd.read_csv('data/pedalboard-plugin.csv', sep=",", index_col=['id', 'name'])

n_labels = 117

definition = TestDefinition(
    KNeighborsClassifier,
    {'n_neighbors': [5, 15, 25], 'algorithm': ['brute'], 'metric': ['hamming']},
    split_x_y
)

metrics = {
    'accuracy': 'accuracy',
    'hit@5': hit_ratio_score_function(5, n_labels),
    'mrr': mrr_score_function(n_labels),
    'mdcg': mdcg_score_function(n_labels),
    #'map':
}

grid = GridSearchCVMultiRefit(definition, random_state=42, number_of_folds=2, metrics=metrics)
grid.fit(data)

print(grid.best_params())
