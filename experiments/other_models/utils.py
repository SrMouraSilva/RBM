import tensorflow as tf

from rbm.util.util import one_hot_encoding


def x_as_one_hot_encoding(x, categories):
    """
    Format x as one hot encoding
    Also add the searched column
    """
    a, b = x.shape
    with tf.Session() as session:
        return session.run(one_hot_encoding(x.values, depth=categories).reshape((-1, categories*b)))

