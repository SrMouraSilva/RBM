import tensorflow as tf
from itertools import product
from pathlib import Path

from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.random_projection import GaussianRandomProjection

from experiments.data.load_data_util import load_data, load_data_categories, load_small_data, load_small_data_categories
from experiments.model_evaluate.evaluate_method.evaluate_method_function import mrr_score_function, mdcg_score_function, \
    hit_ratio_score_function, accuracy, map_score_function, cross_entropy_function
from experiments.model_evaluate.test_definition import TestDefinition
from experiments.model_evaluate.tests_case_evaluator import TestsCaseEvaluator
from experiments.model_evaluate.split_method import split_with_projection_function, split_x_y, \
    split_with_random_matrix_function, split_with_one_hot_encoding_and_projection_function, \
    split_with_one_hot_encoding_function, split_x_y_word2vec_function, split_x_y_normalized_function, \
    split_with_bag_of_words_function, split_with_rbm_encoding_function

##############
# Read data
##############
from rbm.rbmcf import RBMCF

data = load_data()
#data = load_small_data()

# Hidden column
#y_column = None  # None for test all columns

del data['plugin5']
y_column = 1  # plugin2

#del data['plugin2']
#y_column = 3  # 5th pedal (plugin4)



categories = load_data_categories()
#categories = load_small_data_categories()

##############
# Models
##############
n_samples, n_columns = data.shape
projection = GaussianRandomProjection(n_components=50)
n_labels = 117  # Full
#n_labels = 58  # Small


session = tf.Session()
#rbm = RBMCF().load(session, path)

split_methods = [
    #split_x_y,  # Only to kNN
    split_with_one_hot_encoding_function(n_labels),
    #split_with_bag_of_words_function(n_labels),

    # Future
    #split_with_rbm_encoding_function(session, rbm, n_labels)

    # Ignore
    #split_x_y_normalized_function(n_labels),
    #split_with_random_matrix_function((n_columns-1, n_columns-1)),
    #split_with_projection_function(projection),
    #split_with_one_hot_encoding_and_projection_function(projection, n_labels),
    #split_x_y_word2vec_function(),
]

# Grid search params
knn_params = {'n_neighbors': [1, 5, 10, 15, 20, 25, 40, 60, 80, 100], 'algorithm': ['brute'], 'metric': ['hamming']}
mlp_params = {'hidden_layer_sizes': [20, 40, 80, 100], 'max_iter': [800]}
svm_params_rbf = {
    'C': [2e-3, 2e0, 2e3, 2e6, 2e9, 2e12],
    'gamma': [2e-13, 2e-10, 2e-7, 2e-4, 2e-1, 2e2],
    'kernel': ['rbf'],
    'probability': [True]
}
svm_params_linear = {
    'C': [1e-5, 1e-3, 1e-1, 1e1, 1e3, 1e5],
    'kernel': ['linear'],
    'probability': [True]
}
logistic_params = {'multi_class': ['auto'], 'solver': ['liblinear']}

# Models
models_params = [
    #(KNeighborsClassifier, knn_params, 'accuracy'),
    ##(svm.SVC, svm_params_linear, 'accuracy'), #  <-- Run only with one hot encoding
    #(svm.SVC, svm_params_rbf, 'accuracy'),
    (MLPClassifier, mlp_params, 'accuracy'),
    #(LogisticRegression, logistic_params, 'accuracy'),
]

# Generate list
all_grid_elements = []
for model, params, refit in models_params:
    all_grid_elements += [TestDefinition(*execution) for execution in product([model], [params], split_methods, [refit], [y_column])]


##############
# Run
##############
path = Path('evaluate_results/full')
path = Path('evaluate_results/hide_two')
#path = Path('evaluate_results/small')

metrics = {
    'accuracy': accuracy,
    'hit@5': hit_ratio_score_function(5, n_labels),
    #'mrr': mrr_score_function(n_labels),
    'mdcg': mdcg_score_function(n_labels),
    'map@5': map_score_function(5, n_labels, categories),
    #'cross_entropy': cross_entropy_function(),  # Logistic Regression
}

TestsCaseEvaluator(metrics, cv_outer=5, cv_inner=2).run(all_grid_elements, data, path_save=path)

#session.close()
