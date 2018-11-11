import unittest

import numpy as np
import tensorflow as tf


#import pdb; pdb.set_trace()


class TFTest(unittest.TestCase):

    def setUp(self):
        tf.enable_eager_execution()
        self.restart_seed()

    def restart_seed(self):
        tf.set_random_seed(42)
        np.random.seed(42)
