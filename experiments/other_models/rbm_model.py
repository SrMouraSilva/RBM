from typing import Iterable, Iterator

import tensorflow as tf

from experiments.other_models.other_model import OtherModel
from rbm.rbm import RBM


class RBMOtherModel(OtherModel):

    def __init__(self, create_function):
        super().__init__()

        self._session = tf.Session()

        self._rbm = None

        self._current_train = -1
        self._iterator = RBMPersistedIterator(create_function).__iter__()

    @property
    def column(self):
        return self._current_train % 6

    def initialize(self):
        self._current_train += 1

        # Reload the graph and the session
        if self._rbm is not None:
            self._session.close()
            tf.reset_default_graph()

        self._session = tf.Session()

        # Instantiate a new RBM
        self._rbm = self._iterator.__next__(self._session)

    def fit(self, x, y):
        # The model has already trained
        pass

    def predict(self, x):
        x_recommended = self.recommends(x)
        return x_recommended

    def recommends(self, x, column=None):
        """
        Recommends every class with one probability defined
        """
        if column is None:
            column = self.column

        x = self._prepare_x_as_one_hot_encoding(x.copy(), column)

        return self._rbm.P_h_given_v(x.T)

    def _prepare_x_as_one_hot_encoding(self, x, column):
        """
        Format x in expected RBM format (one hot encoding)
        [0 1 0]+[0 0 1] instead [2, 3]

        Also add the searched column
        """
        not_generate_one_hot_element = -1
        x.insert(column, 'y', [not_generate_one_hot_element] * x.shape[0])

        return tf.one_hot(x.values, depth=117).reshape((-1, 117 * 6)).eval(session=self._session)


class RBMPersistedIterator(Iterable):
    """
    Generates rbm instances and load in the disk the persisted status
    """

    def __init__(self, create_function):
        rbm_example = create_function()
        path = rbm_example.__str__().replace('/', '+')

        self.models_path = [
            f'./results/model/kfold={i}+kfold-intern=0+batch_size=10+{path}/rbm.ckpt'
            for i in range(0, 5) for _ in range(0, 6)
        ]
        self.create_function = create_function
        self.current_path_index = 0

        self._session = None

    def __iter__(self) -> Iterator:
        self.current_path_index = 0
        return self

    def __next__(self, session):
        if self.current_path_index >= len(self.models_path):
            raise StopIteration()

        self.current_path_index += 1

        return self._initialize(session)

    def _initialize(self, session):
        rbm: RBM = self.create_function()
        rbm.load(session=session, path=self.current_path)

        return rbm

    @property
    def current_path(self):
        return self.models_path[self.current_path_index]
