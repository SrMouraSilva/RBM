import numpy as np
from gensim.sklearn_api import W2VTransformer

from experiments.other_models.utils import one_hot_encoding


def split_x_y(data, y_column):
    n_samples, n_columns = data.shape

    columns = [f'plugin{i}' for i in range(1, n_columns + 1)]
    train_columns = columns[0:y_column] + columns[y_column + 1:n_columns + 1]
    test_column = f'plugin{y_column + 1}'

    return data[train_columns], data[test_column]


def split_with_random_matrix_function(shape_matrix):
    matrix = np.random.rand(*shape_matrix)

    def split_x_y_with_random_matrix(data, y_column):
        X, y = split_x_y(data, y_column)

        return X @ matrix, y

    return split_x_y_with_random_matrix


def split_with_projection_function(projection):
    def split_x_y_with_projection(data, y_column):
        X, y = split_x_y(data, y_column)

        return projection.fit_transform(X), y

    return split_x_y_with_projection


def split_with_bag_of_words_and_projection_function(projection, n_labels):
    def split_x_y_split_with_bag_of_words_and_projection(data, y_column):
        X, y = split_x_y(data, y_column)
        X = one_hot_encoding(X, n_labels)

        return projection.fit_transform(X), y

    return split_x_y_split_with_bag_of_words_and_projection


def split_with_bag_of_words_function(n_labels):
    def split_x_y_split_with_bag_of_words(data, y_column):
        X, y = split_x_y(data, y_column)
        X = one_hot_encoding(X, n_labels)

        return X, y

    return split_x_y_split_with_bag_of_words


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
