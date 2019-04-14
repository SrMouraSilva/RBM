import tensorflow as tf

from rbm.rbm import RBM
from rbm.sampling.contrastive_divergence import ContrastiveDivergence
from rbm.util.util import bernoulli_sample


class PersistentCD(ContrastiveDivergence):

    def __init__(self, k, batch_size):
        super().__init__(k=k)

        self.v = None
        self.batch_size = batch_size

    def initialize(self, model: RBM) -> None:
        super().initialize(model)

        shape = [model.visible_size, self.batch_size]
        noise = bernoulli_sample(tf.random.uniform(shape))

        self.v = tf.Variable(name='PCD-v', initial_value=noise)

    def __call__(self, v):
        # Batch size can be smaller in last batch size update
        # split self.v in two parts: v_new, v_rest
        # run CD
        # concatenate v_new  v_rest)
        # self.v.assign(concatenate
        v_new = self.v[:, :v.get_shape()[1]]
        v_rest = self.v[:, v.get_shape()[1]:]

        CD = super().__call__

        with tf.name_scope('PCD-{}'.format(self.k)):
            P_h0_given_v0, h0, P_hk_given_vk, vk = CD(v_new)

            self.v = self.v.assign(tf.concat([v_new, v_rest], axis=1))

        return P_h0_given_v0, h0, P_hk_given_vk, self.v[:, :v.get_shape()[1]]

    def __str__(self):
        return f'PCD-{self.k}'
