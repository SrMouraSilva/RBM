from pathlib import Path

import pandas as pd
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.random_projection import GaussianRandomProjection

from experiments.model_evaluate.evaluate_method import mrr_score_function
from experiments.model_evaluate.model_evaluate import ModelEvaluate
from experiments.model_evaluate.split_method import split_with_projection_function, split_x_y, \
    split_with_random_matrix_function, split_with_bag_of_words_and_projection_function, \
    split_with_bag_of_words_function, split_x_y_word2vec_function

##############
# Read data
##############
from experiments.other_models.rbm_trained_model import RBMAlreadyTrainedModel

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
    # KNN
    #  TODO Need select best 'n_neighbors'
    #(KNeighborsClassifier, {'n_neighbors': [1], 'algorithm': ['brute'], 'metric': ['hamming']}, split_x_y),
    #(KNeighborsClassifier, {'n_neighbors': [20], 'algorithm': ['brute'], 'metric': ['hamming']}, split_x_y),

    # SVM
    #  With default database
    #(svm.SVC, {'C': [100.0], 'gamma': [1e-05], 'kernel': ['rbf'], 'probability': [True]}, split_x_y),
    #(svm.SVC, {'C': [1.0],   'gamma':  [0.01], 'kernel': ['rbf'], 'probability': [True]}, split_x_y),

    #  With default database @ confusion matrix
    #(svm.SVC, {'C': [1.0],     'gamma': [0.01], 'kernel':  ['rbf'], 'probability': [True]}, split_with_random_matrix_function((n_columns-1, n_columns-1))),
    #(svm.SVC, {'C': [10000.0], 'gamma': [1e-05], 'kernel': ['rbf'], 'probability': [True]}, split_with_random_matrix_function((n_columns-1, n_columns-1))),

    #  With default database applied gaussian random projection
    #(svm.SVC, {'C': [10000.0], 'gamma': [1e-05], 'kernel': ['rbf'], 'probability': [True]}, split_with_projection_function(projection)),

    #  With bag of words
    #   and kernel linear
    #(svm.SVC, {'C': [1.0], 'kernel': ['linear'], 'probability': [True]}, split_with_bag_of_words_function(n_labels)),

    #  TODO Not search best parameters
    #(svm.SVC, {'C': [10000.0], 'gamma': [1e-05], 'kernel': ['rbf'], 'probability': [True]}, split_with_bag_of_words_function(n_labels)),

    #  With bag of words database applied gaussian random projection
    #(svm.SVC, {'C': [10000.0], 'gamma': [1e-05], 'kernel': ['rbf'], 'probability': [True]}, split_with_bag_of_words_and_projection_function(projection, n_labels)),

    # MLP
    #  Not search best parameters
    #(MLPClassifier, {'hidden_layer_sizes': [5], 'max_iter': [500]}, split_x_y),
    #(MLPClassifier, {'hidden_layer_sizes': [5], 'max_iter': [500]}, split_with_bag_of_words_function(n_labels)),

    #  With bag of words database applied gaussian random projection
    #(MLPClassifier, {'hidden_layer_sizes': [5], 'max_iter': [500]}, split_with_bag_of_words_and_projection_function(projection, n_labels)),

    # LogisticRegression
    #  With default dataset
    #(LogisticRegression, {}, split_x_y),
    #  With bag_of_words
    #(LogisticRegression, {}, split_with_bag_of_words_function(n_labels)),
    #  With bag of words database applied gaussian random projection
    #(LogisticRegression, {}, split_with_bag_of_words_and_projection_function(projection, n_labels)),

    (KNeighborsClassifier, {'n_neighbors': [20], 'algorithm': ['brute'], 'metric': ['hamming']}, split_x_y_word2vec_function()),
    (svm.SVC, {'C': [1.0], 'kernel': ['linear'], 'probability': [True]}, split_x_y_word2vec_function()),
    (svm.SVC, {'C': [10000.0], 'gamma': [1e-05], 'kernel': ['rbf'], 'probability': [True]}, split_x_y_word2vec_function()),
    (LogisticRegression, {}, split_x_y_word2vec_function()),
    (MLPClassifier, {'hidden_layer_sizes': [5], 'max_iter': [500]}, split_x_y_word2vec_function()),
]

##############
# Run
##############
path = Path('evaluate_results')

metrics = {
    'accuracy': 'accuracy',
    #'precision_weighted': 'precision_weighted',
    #'recall_weighted': 'recall_weighted',
    #'f1_weighted': 'f1_weighted',
    'mrr': mrr_score_function(n_labels)
}

ModelEvaluate(metrics).run(models, data, path_save=path)

##############
# RBM
# Not possible execute multiple at same time
##############
models = [
    (RBMAlreadyTrainedModel, {}, split_x_y)
]

#ModelEvaluate(metrics, n_jobs=1).run(models, data, path_save=path)