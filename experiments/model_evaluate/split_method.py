import numpy as np

from experiments.other_models.utils import x_as_one_hot_encoding


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
    def split_x_y_split_with_projection(data, y_column):
        X, y = split_x_y(data, y_column)

        return projection.fit_transform(X), y

    return split_x_y_split_with_projection


def split_with_bag_of_words_and_projection_function(projection, n_labels):
    def split_x_y_split_with_bag_of_words_and_projection(data, y_column):
        X, y = split_x_y(data, y_column)
        X = x_as_one_hot_encoding(X, n_labels)

        return projection.fit_transform(X), y

    return split_x_y_split_with_bag_of_words_and_projection


def split_with_bag_of_words_function(n_labels):
    def split_x_y_split_with_bag_of_words(data, y_column):
        X, y = split_x_y(data, y_column)
        X = x_as_one_hot_encoding(X, n_labels)

        return X, y

    return split_x_y_split_with_bag_of_words
