

class SamplingMethod(object):

    def __init__(self):
        self.model = None

    def initialize(self, model):
        """
        :param model: `RBM` model instance
            rbm-like model implemeting :meth:`rbm.model.gibbs_step` method
        """
        self.model = model
