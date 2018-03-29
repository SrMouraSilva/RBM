
class LearningRate:
    def __init__(self, lr):
        self.base_lr = lr
        self.lr = CustomDefaultDict(lambda: theano.shared(np.array(lr, dtype=config.floatX)))

    def set_individual_lr(self, param, lr):
        self.lr[param].set_value(lr)

    def __mul__(self, other):
        return other

    def __rmul__(self, other):
        return other

    def __call__(self, gradients):
        raise NameError('Should be implemented by inheriting class!')

    def __getstate__(self):
        # Convert defaultdict into a dict
        self.__dict__.update({"lr": CustomDict(self.lr)})
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)

        if type(self.lr) is not CustomDict:
            self.lr = CustomDict()
            for k, v in state['lr'].items():
                self.lr[k] = v

        # Make sure each learning rate have the right dtype
        self.lr = CustomDict({k: theano.shared(v.get_value().astype(config.floatX), name='lr_' + k) for k, v in self.lr.items()})