import numpy as np


def one_hot_encoding(x, depth, dtype=float):
    """
    1-hot encode x with the max value
    """
    if len(x.shape) == 1:
        n_columns = 1
    else:
        n_samples, n_columns = x.shape

    return np.eye(depth, dtype=dtype)[x].reshape((-1, depth*n_columns))


def complete_missing_classes(predictions_with_missing_classes, classes, n_expected_classes, value=0):
    labels = np.array(range(n_expected_classes))
    predictions = predictions_with_missing_classes

    for label in labels:
        if label not in classes:
            predictions = np.insert(predictions, label, value, axis=1)

    return predictions
