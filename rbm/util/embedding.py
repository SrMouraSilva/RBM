import numpy as np
import tensorflow as tf

from rbm.util.util import Σ


def reasonable_visible_bias(data_train):
    """
    Based in Training RBM
    Chapter~8. The initial values of the weights and biases
    """
    total_elements, size_element = data_train.shape

    proportion = Σ(data_train, axis=0) / total_elements
    proportion = proportion.reshape([size_element, 1])

    reasonable = tf.math.log(proportion / (1 - proportion))

    return tf.where(tf.is_inf(reasonable), tf.zeros_like(reasonable), reasonable)


def k_hot_encoding(k: int, elements, n_labels):
    """
    :return: K elements with highest value as one and others as zero
    """
    top_k_index = np.argpartition(elements, -k)[:, -k:]

    return one_hot_encoding(top_k_index, depth=n_labels) \
        .reshape([-1, k, n_labels]) \
        .sum(axis=1, dtype=np.bool)


def one_hot_encoding(x, depth, dtype=np.float32, reshape=True):
    """
    One-hot encoding
    See: https://en.wikipedia.org/wiki/One-hot
    """
    if len(x.shape) == 1:
        n_columns = 1
    else:
        n_samples, n_columns = x.shape

    encoding = tf.keras.utils.to_categorical(x, num_classes=depth, dtype=dtype)
    if reshape:
        return encoding.reshape((-1, depth*n_columns))

    return encoding


def bag_of_words(X, depth, dtype=np.float32):
    return one_hot_encoding(X, depth, dtype=dtype, reshape=False).sum(axis=1)
