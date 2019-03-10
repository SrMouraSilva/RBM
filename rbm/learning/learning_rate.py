from abc import ABCMeta, abstractmethod

from rbm.util.util import Gradient


class LearningRate(metaclass=ABCMeta):

    def __mul__(self, gradient: Gradient):
        return self.calculate(gradient)

    def __rmul__(self, gradient: Gradient):
        return self.__mul__(gradient)

    @abstractmethod
    def calculate(self, dθ):
        pass
