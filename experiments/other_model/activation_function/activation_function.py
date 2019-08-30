from abc import abstractmethod, ABCMeta

import numpy as np


class ActivationFunction(meta=ABCMeta):
    @abstractmethod
    def __call__(self, input_data):
        pass

    @abstractmethod
    def derivative(self, output, doutput):
        pass


class Softmax(ActivationFunction):
    """
    Stable softmax
    """

    def __call__(self, input_data):
        maximum = input_data.max(axis=1)
        maximum_vector = maximum.reshape((-1, 1))

        output = np.exp(input_data - maximum_vector)
        return output / output.sum(axis=1).reshape((-1, 1))

    def derivative(self, output, doutput):
        return output * (doutput - (doutput * output).sum(axis=1).reshape((-1, 1)))
