import tensorflow as tf

from rbm.sampling.sampling_method import SamplingMethod
from rbm.util.util import bernoulli_sample


class ContrastiveDivergence(SamplingMethod):
    """
    :param int k: Number of Gibbs step to do
    """

    def __init__(self, k=1):
        super().__init__()

        self.k = k

    def __call__(self, v0) -> (tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor):
        """
        .. code-block:: python

           P_h0_given_v0 = self.P_h_given_v(v0)
           h0 = bernoulli_sample(p=P_h0_given_v0)

           P_v1_given_h0 = self.P_v_given_h(h0)
           v1 = bernoulli_sample(p=P_v1_given_h0)

           P_h1_given_v1 = self.P_h_given_v(v1)
           h1 = bernoulli_sample(p=P_h1_given_v1)

           P_v2_given_h1 = ...

        :param v0: Visible layer
        :return:
        """
        with tf.name_scope('CD-1'.format(self.k)):
            P_h0_given_v0 = self.model.P_h_given_v(v0)
            h0 = bernoulli_sample(p=P_h0_given_v0)

            def body(i, P_h0_given_v0, h0, v1):
                P_v1_given_h0 = self.model.P_v_given_h(h0)
                v1 = bernoulli_sample(p=P_v1_given_h0)

                P_h1_given_v1 = self.model.P_h_given_v(v1)
                h1 = bernoulli_sample(p=P_h1_given_v1)

                return i + 1, P_h1_given_v1, h1, v1

        with tf.name_scope('CD-{}'.format(self.k)):
            i = tf.constant(0.)
            condition = lambda i, *args: tf.less(i, self.k*1)
            loop_vars = [
                i,
                tf.identity(P_h0_given_v0),
                tf.identity(h0),
                tf.identity(v0)
            ]

            k, P_hk_given_vk, _, vk = tf.while_loop(condition, body, loop_vars, **self.while_params)

        return P_h0_given_v0, h0, P_hk_given_vk, vk

    @property
    def while_params(self):
        return {
            'shape_invariants': None,
            'parallel_iterations': 1,
            'back_prop': False,
            'swap_memory': False,
            'return_same_structure': True
        }

    def __str__(self):
        return f'CD-{self.k}'
