from abc import abstractmethod, ABCMeta

import numpy as np
from tensorflow import Tensor


class EvaluateMethod(metaclass=ABCMeta):

    @abstractmethod
    def evaluate(self, y: np.array, y_predicted: Tensor):
        pass

    def __call__(self, y, y_predicted):
        self.evaluate(y, y_predicted)
