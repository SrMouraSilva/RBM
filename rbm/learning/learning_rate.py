from abc import ABCMeta


class LearningRate(metaclass=ABCMeta):
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    @property
    def η(self):
        """
        :return: :attr:`.learning_rate`
        """
        return self.learning_rate

    @property
    def updates(self):
        return None

    def __mul__(self, other):
        return self.learning_rate * other

    def __rmul__(self, other):
        return other * self.learning_rate
