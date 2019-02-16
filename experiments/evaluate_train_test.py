from pathlib import Path

import pandas as pd
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.random_projection import GaussianRandomProjection

from experiments.model_evaluate.evaluate_method import mrr_score_function
from experiments.model_evaluate.model_evaluate import ModelEvaluate
from experiments.model_evaluate.split_method import split_x_y, \
    split_with_one_hot_encoding_function

##############
# Read data
##############
data = pd.read_csv('data/pedalboard-plugin.csv', sep=",", index_col=['id', 'name'])

##############
# Models
##############
n_samples, n_columns = data.shape
projection = GaussianRandomProjection(n_components=50)
n_labels = 117

# Tuples
# (model, params like grid search, split data method)

models = [
    #(LogisticRegression, {}, split_with_one_hot_encoding_function(n_labels)),
    #(svm.SVC, {'C': [10000.0], 'gamma': [1e-05], 'kernel': ['rbf'], 'probability': [True]}, split_with_one_hot_encoding_function(n_labels)),
    #(KNeighborsClassifier, {'n_neighbors': [15], 'algorithm': ['brute'], 'metric': ['hamming']}, split_x_y),
    (MLPClassifier, {'hidden_layer_sizes': [80], 'max_iter': [500]}, split_with_one_hot_encoding_function(n_labels)),
]

##############
# Run
##############
path = Path('evaluate_results/train-test')

metrics = {
    'accuracy': 'accuracy',
    #'mrr': mrr_score_function(n_labels)
}


ModelEvaluate(metrics).run(models, data, path_save=path)
