import tensorflow as tf

from rbm.model import Model


class SamplingMethod(object):

    def __init__(self):
        self.model = None

    def initialize(self, model: Model) -> None:
        """
        :param model: `RBM` model instance
            rbm-like model implemeting :meth:`rbm.model.gibbs_step` method
        """
        self.model = model

    def __call__(self, v0) -> (tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor):
        """
        :param v0: Visible layer
        :return: Tuple(P_h0_given_v0, h0, P_hk_given_vk, vk)
        """
        pass
