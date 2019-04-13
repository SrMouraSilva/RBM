from itertools import product
from pathlib import Path

from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.random_projection import GaussianRandomProjection

from experiments.data.load_data_util import load_data, load_data_categories
from experiments.model_evaluate.evaluate_method.evaluate_method import mrr_score_function, mdcg_score_function, \
    hit_ratio_score_function, accuracy, map_score_function
from experiments.model_evaluate.model_evaluate import ModelEvaluate, TestDefinition
from experiments.model_evaluate.split_method import split_with_projection_function, split_x_y, \
    split_with_random_matrix_function, split_with_one_hot_encoding_and_projection_function, \
    split_with_one_hot_encoding_function, split_x_y_word2vec_function, split_x_y_normalized_function, \
    split_with_bag_of_words_function

##############
# Read data
##############
data = load_data()
categories = load_data_categories()

##############
# Models
##############
n_samples, n_columns = data.shape
projection = GaussianRandomProjection(n_components=50)
n_labels = 117


split_methods = [
    #split_x_y,
    #split_x_y_normalized_function(n_labels),
    #split_with_random_matrix_function((n_columns-1, n_columns-1)),
    #split_with_projection_function(projection),
    split_with_one_hot_encoding_function(n_labels),
    #split_with_one_hot_encoding_and_projection_function(projection, n_labels),
    #split_x_y_word2vec_function(),
    #split_with_bag_of_words_function(n_labels),
]

# Grid search params
knn_params = {'n_neighbors': [1, 5, 10, 15, 20, 25, 40, 60, 80, 100], 'algorithm': ['brute'], 'metric': ['hamming']}
mlp_params = {'hidden_layer_sizes': [2, 5, 10, 20, 40, 80], 'max_iter': [500]}
svm_params_rbf = {
    'C': [1e-5, 1e-3, 1e-1, 1e1, 1e3, 1e5],
    'gamma': [1e-5, 1e-3, 1e-1, 1e1, 1e3, 1e5, 'scale'],
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
    #(KNeighborsClassifier, knn_params),
    #(svm.SVC, svm_params_linear), #  <-- Run only with one hot encoding
    (svm.SVC, svm_params_rbf),
    #(MLPClassifier, mlp_params),
    #(LogisticRegression, logistic_params),
]

# Generate list
all_grid_elements = []
for model, params in models_params:
    all_grid_elements += [TestDefinition(*execution) for execution in product([model], [params], split_methods)]


##############
# Run
##############
path = Path('evaluate_results')

metrics = {
    'accuracy': accuracy,
    'hit@5': hit_ratio_score_function(5, n_labels),
    'mrr': mrr_score_function(n_labels),
    'mdcg': mdcg_score_function(n_labels),
    'map@5': map_score_function(5, n_labels, categories),
    'map@1': map_score_function(1, n_labels, categories),
}

ModelEvaluate(metrics, cv_outer=5, cv_inner=2).run(all_grid_elements, data, path_save=path)
