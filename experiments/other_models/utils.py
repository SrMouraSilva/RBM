import numpy as np
import tensorflow as tf

from rbm.util.util import one_hot


def one_hot_encoding(x, depth):
    """
    Format x as one hot encoding
    Also add the searched column
    """
    if len(x.shape) == 1:
        n_columns = 1
    else:
        n_samples, n_columns = x.shape

    with tf.Session() as session:
        return session.run(one_hot(x.values, depth=depth).reshape((-1, depth*n_columns)))


def complete_missing_classes(predictions_with_missing_classes, classes, n_expected_classes, value=0):
    labels = np.array(range(n_expected_classes))
    predictions = predictions_with_missing_classes

    for label in labels:
        if label not in classes:
            predictions = np.insert(predictions, label, value, axis=1)

    return predictions
