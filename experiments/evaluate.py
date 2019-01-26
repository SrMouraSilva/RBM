from pathlib import Path

import pandas as pd
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.random_projection import GaussianRandomProjection

from experiments.model_evaluate.model_evaluate import ModelEvaluate
from experiments.model_evaluate.split_method import split_with_projection_function, split_x_y
from experiments.model_evaluate.split_method import split_with_random_matrix_function

##############
# Read data
##############
data = pd.read_csv('data/pedalboard-plugin.csv', sep=",", index_col=['id', 'name'])

##############
# Models
##############
n_samples, n_columns = data.shape
projection = GaussianRandomProjection(n_components=50)

# Tuples
# (model, params like grid search, split data method)

models = [
    (KNeighborsClassifier, {'n_neighbors': [1], 'algorithm': ['brute'], 'metric': ['hamming']}, split_x_y),

    (svm.SVC, {'C': [100.0], 'gamma': [1e-05], 'kernel': ['rbf']}, split_x_y),
    (svm.SVC, {'C': [1.0],   'gamma':  [0.01], 'kernel': ['rbf']}, split_x_y),

    (svm.SVC, {'C': [1.0],     'gamma': [0.01], 'kernel':  ['rbf']}, split_with_random_matrix_function((n_columns-1, n_columns-1))),
    (svm.SVC, {'C': [10000.0], 'gamma': [1e-05], 'kernel': ['rbf']}, split_with_random_matrix_function((n_columns-1, n_columns-1))),

    (svm.SVC, {'C': [10000.0], 'gamma': [1e-05], 'kernel': ['rbf']}, split_with_projection_function(projection)),
]

##############
# Run
##############
path = Path('evaluate_results')
metrics = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']

ModelEvaluate(metrics).run(models, data, path_save=path)
