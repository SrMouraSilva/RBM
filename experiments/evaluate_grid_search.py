from itertools import product
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


split_methods = [
    split_x_y,
    split_with_random_matrix_function((n_columns-1, n_columns-1)),
    split_with_projection_function(projection),
    split_with_bag_of_words_function(n_labels),
    split_with_bag_of_words_and_projection_function(projection, n_labels),
    split_x_y_word2vec_function()
]

# Grid search pparams
knn_params = {'n_neighbors': [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100], 'algorithm': ['brute'], 'metric': ['hamming']}
mlp_params = {'hidden_layer_sizes': [2, 5, 10, 20, 40, 80], 'max_iter': [500]}
svm_params = [{
    'C': [1e-5, 10e-3, 10e-1, 10e1, 10e3, 10e5],
    'gamma': [1e-5, 10e-3, 10e-1, 10e1, 10e3, 10e5, 'scale'],
    'kernel': ['rbf']
}, {
    'C': [1e-5, 10e-3, 10e-1, 10e1, 10e3, 10e5],
    'kernel': ['linear']
}]
logistic_params = {}

# Models
models = [
    (KNeighborsClassifier, knn_params),
    #(svm.SVC, svm_params),
    #(MLPClassifier, mlp_params),
    #(LogisticRegression, logistic_params),
]

# Generate list
all_grid_elements = []
for model, params in models:
    all_grid_elements += list(product([model], [params], split_methods))


##############
# Run
##############
path = Path('evaluate_results/grid-search')

metrics = {
    'accuracy': 'accuracy',
    'precision_weighted': 'precision_weighted',
    'recall_weighted': 'recall_weighted',
    'f1_weighted': 'f1_weighted',
    'mrr': mrr_score_function(n_labels)
}

ModelEvaluate(metrics, cv_outer=5, cv_inner=2).run(models, data, path_save=path)
