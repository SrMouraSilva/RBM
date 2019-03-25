import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import confusion_matrix


def k_hot_encoding(k: int, elements, n_labels):
    """
    :return: K elements with highest value as one and others as zero
    """
    top_k_index = np.argpartition(elements, -k)[:, -k:]

    return one_hot_encoding(top_k_index, depth=n_labels) \
        .reshape([-1, k, n_labels]) \
        .sum(axis=1, dtype=np.bool)


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


def plot_confusion_matrix(title: str, y_originals: list, y_predicts: list, columns_order, columns_names):
    matrix = np.zeros(shape=[117, 117])

    for y, y_predict in zip(y_originals, y_predicts):
        matrix += confusion_matrix(y, y_predict, labels=range(117))

    matrix = pd.DataFrame(matrix)
    matrix = matrix[columns_order]
    matrix = matrix.reindex(columns_order)
    matrix.columns = columns_names
    matrix.index = columns_names

    # Remove y_originals not used
    #matrix = matrix[matrix.sum(axis=1) != 0]
    # Remove y_predicts not used
    #matrix = matrix.T[matrix.T.sum(axis=1) != 0].T

    # Blank not recommended values
    mask = matrix == 0

    ax = sns.heatmap(data=matrix, cmap="YlGnBu", center=1, mask=mask, square=True)
    ax.set_xlabel('predict')
    ax.set_ylabel('test')
    ax.set_title(title)

    return ax
