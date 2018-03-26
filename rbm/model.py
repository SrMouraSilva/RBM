import numpy as np


class Model(object):
    """
    The RBM extensions inherits it
    """

    def __init__(self, random_state=None):
        """
        :param random_state: Numpy random state
        """
        self.random_state = np.random.RandomState() if random_state is None else random_state

