import numpy as np


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
