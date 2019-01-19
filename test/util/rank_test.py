from numpy.testing import assert_almost_equal

from rbm.util.rank import mean_reciprocal_rank
from rbm.util.util import *
from test.tf_test import TFTest


class RankTest(TFTest):

    def test_rank(self):
        tf.enable_eager_execution()

        labels = tf.convert_to_tensor([
         # 1/3 1/2 1/1
            [0, 0,  1],
         # 1/3 1/2 1/1
            [1, 0,  0],
        ])
        values = tf.convert_to_tensor([
            [0.1, .2, .3],
            [0.1, .2, .4],
        ])

        assert_almost_equal((1 + 1/3)/2, mean_reciprocal_rank(labels, values).numpy())
