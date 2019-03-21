import numpy as np
import tensorflow as tf


def one_hot_encoding(x, depth, dtype=np.float32, reshape=True):
    if len(x.shape) == 1:
        n_columns = 1
    else:
        n_samples, n_columns = x.shape

    encoding = tf.keras.utils.to_categorical(x, num_classes=depth, dtype=dtype)
    if reshape:
        return encoding.reshape((-1, depth*n_columns))

    return encoding


def complete_missing_classes(predictions_with_missing_classes, classes, n_expected_classes, value=0):
    labels = np.array(range(n_expected_classes))
    predictions = predictions_with_missing_classes

    for label in labels:
        if label not in classes:
            predictions = np.insert(predictions, label, value, axis=1)

    return predictions
