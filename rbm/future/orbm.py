from abc import ABCMeta

import numpy as np
import tensorflow as tf

from rbm.rbm import RBM
from rbm.train.persistent import Persistent
from rbm.util.util import Σ, softplus, σ, bernoulli_sample, mean, gradient, Gradient, square, softmax


class oRBMPenalty(metaclass=ABCMeta):
    """
    NOT IMPLEMENTED yet :(
    If you need, use
    https://github.com/MarcCote/iRBM/blob/master/iRBM/models/orbm.py

    .. note::

        "In our experiments, as we wanted the filters of each unit to be the dominating
        factor in a unit being selected, we parametrized it as :math:`\\beta_i = \\beta \cdot soft_{+}(b^h_i)`,
        where :math:`\\beta` is a global hyper-parameter (critically, as we’ll discuss later,
        this hyper-parameter doesnt actually require tuning and a generic value for it works fine).

        Intuitively, the penalty term acts as a form of regularization since it forces the model to
        avoid using more hidden units than needed, prioritizing smaller networks."
        -- :cite:`cote2016infinite`

    """

    def __init__(self):
        self.model = None
        self.beta = None  # https://github.com/MarcCote/iRBM/blob/master/iRBM/models/orbm.py#L22

    @property
    def value(self):
        pass

    def __sub__(self, other):
        return other - self.value

    def __rsub__(self, other):
        return self.__sub__(other)


class ConstantPenalty(oRBMPenalty):
    """
    Does not interfere with the result
    """

    @property
    def value(self):
        return tf.constant(0)


class SoftPlusBiPenalty(oRBMPenalty):

    @property
    def value(self):
        """
        :getter tf.constant: :math:`\\beta_i = \\beta \cdot soft_{+} (b_i^h)`
        Returns :math:`\\boldsymbol{\\beta} = [\\beta_1, ..., \\beta_z]`
        """
        z = self.model.z
        b_split = self.model.b[:z]

        return self.beta @ softplus(b_split)


class SoftPlusZeroPenalty(oRBMPenalty):

    @property
    def value(self):
        """
        :getter tf.constant: :math:`\\beta_i = \\beta \cdot soft_{+} (0)`
        Returns :math:`\\boldsymbol{\\beta} = [\\beta_1, ..., \\beta_z]`
        """
        return self.beta @ softplus(0)


class oRBM(RBM, Persistent):
    """
    "Ordered Restricted Boltzmann Machine (oRBM)
    is a variant of the RBM where the hidden units :math:`\\boldsymbol{h}`
    are ordered from left to right, with this order being taken into account
    by the energy function" -- :cite:`cote2016infinite`.

    Only the first :math:`z`
    elements of the hidden layer will be considered.

    :param visible_size: ``D`` Size of the visible layer
    :param hidden_size: ``K`` Size of the hidden layer
    :param penalty: Intuitively, the penalty term acts as a form of
        regularization since it forces the model to
        avoid using more hidden units than needed, prioritizing smaller networks :cite:`cote2016infinite`.
    :param sampling_method: CD or PCD
    """

    def __init__(self, visible_size: int, hidden_size: int, penalty: oRBMPenalty=None, *args, **kwargs):
        self.penalty = penalty if penalty is not None else ConstantPenalty()

        super(oRBM, self).__init__(visible_size, hidden_size, *args, **kwargs)

    def setup(self):
        """
        Initialize objects related to the RBM, like the :attr:`~rbm.rbm.RBM.sampling_method`
        and the :attr:`~rbm.rbm.RBM.regularization`
        """
        super(oRBM, self).setup()

        self.penalty.model = self

    def E(self, v, h, z):
        """
        Energy function

        .. math::

            E(\\boldsymbol{v}, \\boldsymbol{h}, z) = - \\boldsymbol{v}^T\\boldsymbol{b}^v
                - \sum_{i=1}^z \\left( h_i (\\boldsymbol{W}_{i\cdot} + b_i^h) - \\beta_i \\right)

                = - \\boldsymbol{h}_{1:z}^T\\boldsymbol{W}_{\cdot 1:z}\\boldsymbol{v}
                  - \\boldsymbol{v}^T\\boldsymbol{b}^v - \\boldsymbol{h}_{1:z}^T\\boldsymbol{b}_{1:z}^h
                  - \sum{\\boldsymbol{\\beta}}

        :param v: :math:`\\boldsymbol{v}` Visible layer
        :param h: :math:`\\boldsymbol{h}` Hidden layer
        :param z: :math:`z` Number of selected hidden units that are active.
                  I.e., only the first :math:`z` will be used (:math:`[h_1, h_2, ..., h_z]`)

        :return:
        """
        with tf.name_scope('energy'):
            # Use the first z-th elements
            # [columns, lines]
            h_slice = h[:, 0:z]
            W_slice = self.W[0:z, :]

            return h_slice.T @ W_slice @ v - (v.T @ self.b_v) - (h_slice.T @ self.b_h) - Σ(self.penalty.value)

    def F(self, v):
        """
        The :math:`F(\\boldsymbol{v})` is the free energy function

        .. math::

            F(\\boldsymbol{v})
                = \sum_{z=1}^{K} F(\\boldsymbol{v}, z)
                = - ln \\left(\sum_{z=1}^{K} e^{-F(\\boldsymbol{v}, z)} \\right)

        .. math::

                = - ln \\left( \sum_{z=1}^{K} e^{ - \\left[
                    - \\boldsymbol{v}^T\\boldsymbol{b}^v
                    - \sum_{i=1}^{z} \\left(
                        soft_{+}(\\boldsymbol{W}_{i\cdot} \\boldsymbol{v} + b_i^h) - \\beta_i
                    \\right)
                \\right]
                } \\right)

                = - ln \\left(
                    e^{ \\boldsymbol{v}^T\\boldsymbol{b}^v}
                    \sum_{z=1}^{K} e^{
                    \sum_{i=1}^{z} \\left(
                        soft_{+}(\\boldsymbol{W}_{i\cdot} \\boldsymbol{v} + b_i^h) - \\beta_i
                    \\right)
                } \\right)

                = - \\boldsymbol{v}^T\\boldsymbol{b}^v
                  - ln \\left(
                    \sum_{z=1}^{K} e^{
                    \sum_{i=1}^{z} \\left(
                        soft_{+}(\\boldsymbol{W}_{i\cdot} \\boldsymbol{v} + b_i^h) - \\beta_i
                    \\right)
                } \\right)

                = - \\boldsymbol{v}^T\\boldsymbol{b}^v
                  - \\text{logsumexp} \\left(
                    \\text{cumsum} \\left(
                    soft_{+}(\\boldsymbol{W}_{i\cdot} \\boldsymbol{v} + b_i^h) - \\beta_i
                  \\right)
                \\right)

        Where

        * ``K`` is the :attr:`.oRBM.hidden_size` (cardinality of the hidden layer);
        * :math:`F(\\boldsymbol{v}, z)` is the free energy function with z not marginalized (see :meth:`.oRBM.Fvz`);
        * :math:`z` is the effective number of hidden units participating to the energy.

        :param v: :math:`\\boldsymbol{v}` Visible layer
        :return: :math:`F(\\boldsymbol{v})`
        """
        with tf.name_scope('free_energy'):
            logsumexp = lambda x: tf.reduce_logsumexp(x, axis=1)
            # Cumsum over z
            cumsum = lambda x: tf.cumsum(x, axis=1)

            return -(v.T @ self.b_v) - logsumexp(cumsum(softplus(self.W @ v + self.b_h) - self.penalty))

    def _log_z_given_v(self, v):
        # Cumsum over z
        cumsum = lambda x: tf.cumsum(x, axis=1)

        return cumsum(softplus(self.W @ v + self.b_h) - self.penalty)

    def Fvz(self, v, z):
        """
        The :math:`F(\\boldsymbol{v}, z)` is the free energy function, where z is not marginalized

        .. math::

            F(\\boldsymbol{v}, z) = - \\boldsymbol{v}^T\\boldsymbol{b}^v
                                 - \sum_{i=1}^{z}
                                 \\left(
                                    soft_{+}(\\boldsymbol{W}_{i\cdot} \\boldsymbol{v} + b_i^h)
                                    - \\beta_i
                                 \\right)

        For :math:`soft_{+}(x)` see :meth:`~rbm.util.util.softplus`

        :param v: :math:`\\boldsymbol{v}` Visible layer
        :param z: :math:`z` is the effective number of hidden units participating to the energy.
        """
        W_slice = self.W[0:z, :]
        b_h_slice = self.b_h[:, 0:z]

        return -(v.T @ self.b_v) - Σ(softplus(W_slice @ v + b_h_slice) - self.penalty)

    def P_z_given_v(self, v):
        """

        .. math::

            P(z|\\boldsymbol{v})
                = \\frac{P(\\boldsymbol{v}, z)}{P(\\boldsymbol{v})}
                = \\frac{e^{-F(\\boldsymbol{v}, z)}}
                       {\sum_{z'=1}^K e^{-F(\\boldsymbol{v}, z')}}
                = \\text{softmax}(-F(\\boldsymbol{v}, z))

        Simplifying

        .. math::

                = \\frac{
                    e^{
                        - \\boldsymbol{v}^T\\boldsymbol{b}^v
                        - \sum_{i=1}^{z} \\left(
                        soft_{+}(\\boldsymbol{W}_{i\cdot} \\boldsymbol{v} + b_i^h) - \\beta_i
                        \\right)
                    }
                }{
                    \sum_{z'=1}^K e^{
                        - \\boldsymbol{v}^T\\boldsymbol{b}^v
                        - \sum_{i=1}^{z'} \\left(
                        soft_{+}(\\boldsymbol{W}_{i\cdot} \\boldsymbol{v} + b_i^h) - \\beta_i
                        \\right)
                    }
                }
                = \\frac{
                    e^ { - \\boldsymbol{v}^T\\boldsymbol{b}^v }
                    e^{
                        - \sum_{i=1}^{z} \\left(
                        soft_{+}(\\boldsymbol{W}_{i\cdot} \\boldsymbol{v} + b_i^h) - \\beta_i
                        \\right)
                    }
                }{
                    e^ { - \\boldsymbol{v}^T\\boldsymbol{b}^v }
                    \sum_{z'=1}^K e^{
                        - \sum_{i=1}^{z'} \\left(
                        soft_{+}(\\boldsymbol{W}_{i\cdot} \\boldsymbol{v} + b_i^h) - \\beta_i
                        \\right)
                    }
                }

                = \\frac{
                    e^{
                        - \sum_{i=1}^{z} \\left(
                        soft_{+}(\\boldsymbol{W}_{i\cdot} \\boldsymbol{v} + b_i^h) - \\beta_i
                        \\right)
                    }
                }{
                    \sum_{z'=1}^K e^{
                        - \sum_{i=1}^{z'} \\left(
                        soft_{+}(\\boldsymbol{W}_{i\cdot} \\boldsymbol{v} + b_i^h) - \\beta_i
                        \\right)
                    }
                }

                = \\text{softmax}\\left[
                    - \sum_{i=1}^{z} \\left(
                    soft_{+}(\\boldsymbol{W}_{i\cdot} \\boldsymbol{v} + b_i^h) - \\beta_i
                    \\right)
                \\right]

        :param v: :math:`\\boldsymbol{v}` Visible layer
        :return: :math:`P(z|\\boldsymbol{v})`
        """
        with tf.name_scope('P_z_given_v'):
            log_z_given_v = self._log_z_given_v(v)
            return softmax(log_z_given_v)  # If 2D, softmax is perform along axis=1.

    def sample_h_given_v(self, v):
        """
        With the :math:`P(\\boldsymbol{h} = 1|\\boldsymbol{v})` (obtained from :meth:`.RBM.P_h_given_v`), is generated
        a sample of :math:`\\boldsymbol{h}` with the Bernoulli distribution.

        :param v: :math:`\\boldsymbol{v}` Visible layer
        :return: The hidden layer :math:`\\boldsymbol{h}` sampled from :math:`\\boldsymbol{v}`
        """
        with tf.name_scope('sample_h_given_v'):
            z_mask = self.sample_zmask_given_v(v)
            probabilities = self.P_h_given_v(v)
            # Needs to reshape because right now Theano GPU's multinomial supports only pvals.ndim==2 and n==1.
            probabilities = tf.reshape(probabilities, (2, self.visible_size * self.hidden_size)).T

            h_sample = tf.multinomial(n=1, pvals=probabilities, dtype=tf.float32)
            h_sample = h_sample @ np.array([0, 1], dtype=tf.float32)

            # Needs to reshape because right now Theano GPU's multinomial supports only pvals.ndim==2 and n==1.
            h_sample = tf.reshape(h_sample, (v.shape[0], self.hidden_size))
            return z_mask * h_sample

    def P_h_given_v(self, v):
        # The real P_h_given_v = σ(self.W @ v + self.b_h)
        with tf.name_scope('P_h_given_v'):
            # Wx_plusb = z_mask * (self.W @ v + self.b_h)
            prob_h = σ(self.W @ v + self.b_h)
            prob_h_nil = σ(-(self.W @ v + self.b_h))

            return tf.stack(prob_h_nil, prob_h)

    def sample_zmask_given_v(self, v):
        cumsum = lambda x: tf.cumsum(x, axis=1)
        probabilities = self.P_z_given_v(v)
        p = self.theano_rng.multinomial(pvals=probabilities, dtype=tf.float32)
        return tf.cumsum(p[:, ::-1], axis=1)[:, ::-1]
