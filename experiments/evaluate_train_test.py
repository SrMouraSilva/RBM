from pathlib import Path

import pandas as pd
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.random_projection import GaussianRandomProjection

from experiments.model_evaluate.evaluate_method.evaluate_method import hit_ratio_score_function, mdcg_score_function, mrr_score_function
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
    # k-NN
    # - Accuracy
    #(KNeighborsClassifier, {'n_neighbors': [15], 'algorithm': ['brute'], 'metric': ['hamming']}, split_x_y),
    # - Hit@5
    #(KNeighborsClassifier, {'n_neighbors': [40], 'algorithm': ['brute'], 'metric': ['hamming']}, split_x_y),
    # - MRR, NDGC
    #(KNeighborsClassifier, {'n_neighbors': [60], 'algorithm': ['brute'], 'metric': ['hamming']}, split_x_y),
    # - MAP
    # ?

    # LogisticRegression
    # - Accuracy, Hit@5, MRR, NDGC, MAE
    #(LogisticRegression, {}, split_with_one_hot_encoding_function(n_labels)),

    # SVM
    # - Accuracy, NDGC
    #(svm.SVC, {'C': [10.0], 'gamma': ['scale'], 'kernel': ['rbf'], 'probability': [True]}, split_with_one_hot_encoding_function(n_labels)),
    # - Hit@5
    #(svm.SVC, {'C': [1000.0], 'gamma': ['scale'], 'kernel': ['rbf'], 'probability': [True]}, split_with_one_hot_encoding_function(n_labels)),
    # - MRR
    #(svm.SVC, {'C': [10.0], 'gamma': [0.1], 'kernel': ['rbf'], 'probability': [True]}, split_with_one_hot_encoding_function(n_labels)),
    # - MAP

    # MLP
    # - Accuracy, Hit@5, MRR, NDGC
    (MLPClassifier, {'hidden_layer_sizes': [80], 'max_iter': [500]}, split_with_one_hot_encoding_function(n_labels)),
    # - MAP
]

##############
# Run
##############
path = Path('evaluate_results/train-test')

metrics = {
    'accuracy': 'accuracy',
    'hit@5': hit_ratio_score_function(5, n_labels),
    'mrr': mrr_score_function(n_labels),
    'mdcg': mdcg_score_function(n_labels),
    #'map':
}


ModelEvaluate(metrics).run(models, data, path_save=path)
