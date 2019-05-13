from typing import Callable, Tuple

import numpy as np
import tensorflow as tf
from gensim.sklearn_api import W2VTransformer

from rbm.rbm import RBM
from rbm.util.embedding import one_hot_encoding, bag_of_words

SplitFunction = Callable[[np.ndarray, int], Tuple[np.ndarray, np.ndarray]]


def split_x_y(data, y_column):
    n_samples, n_columns = data.shape

    columns = data.columns.tolist()
    train_columns = columns[0:y_column] + columns[y_column + 1:n_columns + 1]
    test_column = columns[y_column]

    return data[train_columns], data[test_column]


def split_x_y_normalized_function(n_labels):
    n_labels = n_labels * 1.

    def split_x_y_normalized(data, y_column):
        X, y = split_x_y(data, y_column)

        return X/n_labels, y

    return split_x_y_normalized


def split_with_random_matrix_function(shape_matrix):
    matrix = np.random.rand(*shape_matrix)

    def split_x_y_with_random_matrix(data, y_column):
        X, y = split_x_y(data, y_column)

        return X @ matrix, y

    return split_x_y_with_random_matrix


def split_with_bag_of_words_function(n_labels):
    def split_x_y_with_bag_of_words(data, y_column):
        X, y = split_x_y(data, y_column)
        X = one_hot_encoding(X, n_labels, reshape=False)

        x_bag_of_words = bag_of_words(X, n_labels)
        x_bag_of_words = x_bag_of_words / x_bag_of_words.sum(axis=1).reshape((-1, 1))

        return x_bag_of_words, y

    return split_x_y_with_bag_of_words


def split_with_projection_function(projection):
    def split_x_y_with_projection(data, y_column):
        X, y = split_x_y(data, y_column)

        return projection.fit_transform(X), y

    return split_x_y_with_projection


def split_with_one_hot_encoding_function(n_labels):
    def split_x_y_split_with_one_hot_encoding(data, y_column):
        X, y = split_x_y(data, y_column)
        X = one_hot_encoding(X, n_labels)

        return X, y

    return split_x_y_split_with_one_hot_encoding


def split_with_one_hot_encoding_and_projection_function(projection, n_labels):
    function = split_with_one_hot_encoding_function(n_labels)

    def split_x_y_split_with_one_hot_encoding_and_projection(data, y_column):
        X, y = function(data, y_column)

        return projection.fit_transform(X), y

    return split_x_y_split_with_one_hot_encoding_and_projection


def split_x_y_word2vec_function(size=10, min_count=1, seed=42):
    def split_x_y_word2vec(data, y_column):
        model = W2VTransformer(size=size, min_count=min_count, seed=seed)

        X, y = split_x_y(data, y_column)
        # DataFrame -> ndarray
        X = X.copy().astype(str).values

        n_samples, n_columns = X.shape

        wordvecs = model.fit(X.tolist()).transform(X.reshape(-1).tolist())

        return wordvecs.reshape(n_samples, n_columns*size), y

    return split_x_y_word2vec


def split_with_rbm_encoding_function(session: tf.Session, rbm: RBM, n_labels):
    def split_with_rbm_encoding(data, y_column):
        data = data.values

        y_starts = n_labels * y_column
        y_ends = n_labels * (y_column+1)

        X = one_hot_encoding(data, depth=n_labels)
        X[:, y_starts:y_ends] = 0

        y = data[y_column]

        return rbm.P_h_given_v(X).run(session=session), y

    return split_with_rbm_encoding
