
class Regularization(object):
    def __init__(self, decay):
        self.decay = decay
        self.parameter = None

    def initialize(self, parameter):
        self.parameter = parameter

    @property
    def value(self):
        return self(self.parameter)

    def __call__(self, param):
        raise NameError('Should be implemented by subclasses!')

    def __add__(self, other):
        return self.value + other

    def __radd__(self, other):
        return other + self.value


class NoRegularization(Regularization):
    def __init__(self):
        Regularization.__init__(self, 0.0)

    def __call__(self, param):
        return 0.0


class L1Regularization(Regularization):
    def __init__(self, decay):
        Regularization.__init__(self, decay)

    def __call__(self, param):
        return self.decay * abs(param).sum()


class L2Regularization(Regularization):
    def __init__(self, decay):
        Regularization.__init__(self, decay)

    def __call__(self, param):
        return 2*self.decay * (param**2).sum()
