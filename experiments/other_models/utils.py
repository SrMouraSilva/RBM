import tensorflow as tf
import numpy as np


def one_hot_encoding(x, depth, dtype=np.float32):
    return tf.keras.utils.to_categorical(x, num_classes=depth, dtype=dtype)


def complete_missing_classes(predictions_with_missing_classes, classes, n_expected_classes, value=0):
    labels = np.array(range(n_expected_classes))
    predictions = predictions_with_missing_classes

    for label in labels:
        if label not in classes:
            predictions = np.insert(predictions, label, value, axis=1)

    return predictions
