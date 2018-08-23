from abc import ABCMeta

import tensorflow as tf

from rbm.rbm import RBM
from rbm.sampling.sampling_method import SamplingMethod
from rbm.train.persistent import Persistent
from rbm.util.util import Σ, softplus, σ, bernoulli_sample, mean, gradient, Gradient, square


class oRBMPenalty(metaclass=ABCMeta):
    """
    "In our experiments, as we wanted the filters of each unit to be the dominating
    factor in a unit being selected, we parametrized it as :math:`\\beta_i = \\beta \cdot soft+(b^h_i)`,
    where :math:`\\beta` is a global hyper-parameter (critically, as we’ll discuss later,
    this hyper-parameter doesnt actually require tuning and a generic value for it works fine).

    Intuitively, the penalty term acts as a form of regularization since it forces the model to
    avoid using more hidden units than needed, prioritizing smaller networks."  :cite:`cote2016infinite`
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
    by the energy function" :cite:`cote2016infinite`. Only the first :math:`z`
    elements of the hidden layer will be considered.

    :param visible_size: ``D`` Size of the visible layer
    :param hidden_size: ``K`` Size of the hidden layer
    :param penalty: Intuitively, the penalty term acts as a form of
        regularization since it forces the model to
        avoid using more hidden units than needed, prioritizing smaller networks :cite:`cote2016infinite`.
    :param sampling_method: CD or PCD
    """

    def __init__(self, visible_size: int, hidden_size: int, penalty: oRBMPenalty=None, sampling_method: SamplingMethod=None, *args, **kwargs):
        super(oRBM, self).__init__(*args, **kwargs)

        self.penalty = penalty if sampling_method is not None else ConstantPenalty()

        self.θ = [self.W, self.b_h, self.b_v, z]

        self.setup()

    @property
    def parameters(self):
        """
        .. math: \Theta = \{\mathbf{W}, \mathbf{b}^V, \mathbf{b}^h\, FIXME}

        :return:
        """
        return self.θ

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

            F(\\boldsymbol{v}) = - ln \\left(\sum_{z=1}^{K} e^{F(\\boldsymbol{v}, z)} \\right)

        Where

        * ``K`` is the :attr:`.oRBM.hidden_size` (cardinality of the hidden layer);
        * :math:`F(\\boldsymbol{v}, z)` is the free energy function with z not marginalized (see :meth:`.oRBM.Fvz`);
        * :math:`z` is the effective number of hidden units participating to the energy.

        :param v: :math:`\\boldsymbol{v}` Visible layer
        :return: :math:`F(\\boldsymbol{v})`
        """
        with tf.name_scope('free_energy'):
            # Optimization: not uses F(v, z)
            return -(v.T @ self.b_v) - tf.reduce_logsumexp(self._log_z_given_v(v), axis=1)

    def _log_z_given_v(self, v):
        """
        Pq o nome do método tem log?
        """
        energies = softplus(self.W @ v + self.b_h) - self.penalty
        # Cumulative sum for calc all possibles of z: z \in {1, ..., K}
        return tf.cumsum(energies, axis=1)

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
        return -(v.T @ self.b_v) - Σ(softplus(self.W @ v + self.b_h) - self.penalty)

    def sample_h_given_v(self, v):
        """
        With the :math:`P(\\boldsymbol{h} = 1|\\boldsymbol{v})` (obtained from :meth:`.RBM.P_h_given_v`), is generated
        a sample of :math:`\\boldsymbol{h}` with the Bernoulli distribution.

        :param v: :math:`\\boldsymbol{v}` Visible layer
        :return: The hidden layer :math:`\\boldsymbol{h}` sampled from :math:`\\boldsymbol{v}`
        """
        with tf.name_scope('sample_h_given_v'):
            h_mean = self.P_h_given_v(v)
            h_sample = bernoulli_sample(p=h_mean)

            return h_sample

            hidden_size = self.b.shape[0]

            z_mask = self.sample_zmask_given_v(v)

            # Wx_plusb = z_mask * (self.W @ v + self.b_h)
            prob_h = self.P_h_given_v(v)
            # mesmo que prob_h, mas as coisas sendo negativas
            prob_h_nil = σ(-(self.W @ v + self.b_h))

            prob = T.stack(prob_h_nil, prob_h)
            # Needs to reshape because right now Theano GPU's multinomial supports only pvals.ndim==2 and n==1.
            prob = T.reshape(prob, (2, v.shape[0] * hidden_size)).T

            h_sample = self.theano_rng.multinomial(n=1, pvals=prob, dtype=theano.config.floatX)
            h_sample = T.dot(h_sample, np.array([0, 1], dtype=theano.config.floatX))

            # Needs to reshape because right now Theano GPU's multinomial supports only pvals.ndim==2 and n==1.
            h_sample = T.reshape(h_sample, (v.shape[0], hidden_size))
            return z_mask * h_sample

    def learn(self, v):
        with tf.name_scope('calculate_parameters'):
            updates = self.calculate_parameters_updates(v)

        assignments = []

        for parameter, update in zip(self.parameters, updates):
            parameter_name = parameter.op.name.split('/')[-1]
            with tf.name_scope('assigns/assign_' + parameter_name):
                assignments.append(parameter.assign(update))

        return assignments

    def calculate_parameters_updates(self, v) -> []:
        """
        There are the gradient descent for RBM:

        .. math:: \\nabla_{\\theta} F(\\theta, \mathcal{D}) =
                    \\frac{1}{N} \\underbrace{
                                    \sum_{n=1}^{N} \\nabla_{\\theta} F(\mathbf{v}_n)
                                 }_\\text{Positive phase}
                               - \\underbrace{
                                    \sum_{\mathbf{v}' \in \mathcal{V}} \\nabla_{\\theta} F(\mathbf{v}')
                                 }_\\text{Negative phase}

        where

        * :math:`\mathcal{D}`: A set of N examples. :math:`\mathcal{D} = \{\mathbf{v}_n\}_{n=1}^N`
        * :math:`\mathcal{V}`: All possibilities for the visible layer (:math:`2^D` possibilities).
                 with :math:`D` = size of the visible layer
        * :math:`\\nabla_{\mathbf{W}} F(\mathbf{v})                 \
                    = \mathbb{E}[\mathbf{h}|\mathbf{v}]\mathbf{v}^T \
                    = - \mathbf{\hat{h}}(\mathbf{v})\mathbf{v}^T`
        * :math:`\\nabla_{\mathbf{b}^h} F(\mathbf{v})   \
                    = \mathbb{E}[\mathbf{h}|\mathbf{v}] \
                    = - \mathbf{\hat{h}}(\mathbf{v})`
        * :math:`\\nabla_{\mathbf{b}^v} F(\mathbf{v})   \
                    = - \mathbf{v}`
        * :math:`\mathbf{\hat{h}}(\mathbf{v}) = \\sigma({\mathbf{Wv} + \mathbf{b}^h})`

        But the negative phase are intractable. Then will

        .. note::

            "approximate the expectation under :math:`P(\mathbf{v})`
            with an average of S samples :math:`\mathcal{S} = \{\mathbf{\hat{v}}\}_{s=1}^S`
            draw from :math:`P(\mathbf{v})` i.e. the model."
            -- Infinite RBM

        .. math:: \\nabla_{\\theta} F(\\theta, \mathcal{D}) \\approx
                    \\underbrace{
                        \\frac{1}{N} \sum_{n=1}^{N} \\nabla_{\\theta} F(\mathbf{v}_n)
                    }_\\text{Positive phase}
                  - \\underbrace{
                        \\frac{1}{S} \sum_{n=1}^{S} \\nabla F(\mathbf{\hat{v}}_s)
                    }_\\text{Negative phase}

        where

        * :math:`\mathbf{\hat{v}}_i`: The i-th sample generated

        :param v: `\mathbf{v}` Array visible layer. A mini-batch of :math:`\mathcal{D}`
        :return: Theano variables updated, like the params :math:`\sigma` with the new value
        """
        F = lambda v: self.F(v)
        CD = self.sampling_method
        θ = self.θ
        Ln = self.regularization
        η = self.learning_rate

        # Contrastive divergence
        with tf.name_scope('samples'):
            samples = CD(v)

        # [Expected] negative log-likelihood + Regularization
        with tf.name_scope('cost'):
            error = mean(F(v)) - mean(F(samples))
            cost = error + Ln

            tf.summary.scalar('Ln', 0 + Ln)
            tf.summary.scalar('cost', cost)
            tf.summary.scalar('MSE', mean(square(error)))

        # Gradients (use automatic differentiation)
        # We must not compute the gradient through the gibbs sampling, i.e. use consider_constant
        gradients = gradient(cost, wrt=θ, consider_constant=[samples])

        # Updates parameters
        parameters = []
        for dθ, parameter in zip(gradients, θ):
            dθ = Gradient(dθ, wrt=parameter)

            with tf.name_scope('calculate_parameters/calculate_' + parameter.op.name.split('/')[-1]):
                parameters.append(parameter - η * dθ)

        return parameters
