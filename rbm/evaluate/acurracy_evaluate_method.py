import numpy as np
import tensorflow as tf

from tensorflow import Tensor
from tensorflow.python.ops.gen_bitwise_ops import bitwise_and

from rbm.evaluate.evaluate_method import EvaluateMethod
from rbm.util.util import count_equals_array


class AccuracyEvaluateMethod(EvaluateMethod):
    """
    Given y and y_expected, count all rows in y_expected that are equals in y.
    If y_expected contains more than one activated element (ex: two recommendations for one row)
    the equality with y will consider if one of the activated in y_expected are equals with y
    """

    def evaluate(self, y: np.array, y_predicted: Tensor):
        total_of_elements = y.shape[0]

        y = y.astype(np.int32)
        y_predicted = y_predicted.cast(tf.int32)

        total_equals = count_equals_array(bitwise_and(y, y_predicted), y)

        return total_equals / total_of_elements
