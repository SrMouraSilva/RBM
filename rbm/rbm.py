from collections import OrderedDict

import numpy as np
import theano

from rbm.model import Model
from rbm.sampling.contrastive_divergence import ContrastiveDivergence
from rbm.util.util import σ, softplus, Σ, mean, gradient_descent, binomial, Gradient


class RBM(Model):
    """
    :param input_size: ``D`` Size of the visible layer
    :param hidden_size: ``K`` Size of the hidden layer
    :param SamplingMethod sampling_method: CD or PCD
    """

    def __init__(self, input_size, hidden_size, sampling_method=None, *args, **kwargs):
        super(RBM, self).__init__(*args, **kwargs)

        self.input_size = input_size
        self.hidden_size = hidden_size

        # TODO - Talvez as dimenções estejam trocadas
        self.W = theano.shared(value=np.zeros((self.hidden_size, self.input_size), dtype=theano.config.floatX),
                               name='W')
        self.b_h = theano.shared(value=np.zeros(self.hidden_size, dtype=theano.config.floatX), name='b_h')
        self.b_v = theano.shared(value=np.zeros(self.input_size, dtype=theano.config.floatX), name='b_v')

        self.θ = [self.W, self.b_h, self.b_v]

        self.sampling_method = sampling_method if sampling_method is not None else ContrastiveDivergence()

        self.setup()

    @property
    def parameters(self):
        """
        .. math: θ = \{\mathbf{W}, \mathbf{b}^V, \mathbf{b}^h\}

        :return:
        """
        return self.θ

    def setup(self):
        """
        Initialize the weight matrix (:attr:`~rbm.rbm.RBM.W`)
        """
        self.W = 1e-2 * self.random_state.randn(self.hidden_size, self.input_size).astype(dtype=np.float64)

    def E(self, v, h):
        """
        Energy function

        .. math::

            E(\mathbf{v}, \mathbf{h}) = - \mathbf{h}^T\mathbf{W}\mathbf{v} - \mathbf{v}^T\mathbf{b}^v - \mathbf{h}^T\mathbf{b}^h

        :param v: Visible layer
        :param h: Hidden layer

        :return:
        """
        return h.T @ self.W @ v - (v.T @ self.b_v) - (h.T @ self.b_h)

    def F(self, v):
        """
        The :math:`F(\mathbf{v})` is the free energy function

        .. math::

            F(\mathbf{v}) = - \mathbf{v}^T\mathbf{b}^v - \sum_{i=1}^{K} soft_{+}(\mathbf{W}_{i\cdot} \mathbf{v} + b_i^h)

        Where ``K`` is the :attr:`~rbm.rbm.RBM.hidden_size` (cardinality of the hidden layer)

        For :math:`soft_{+}(x)` see :meth:`~util.util.softplus`

        :param v: Visible layer
        :return: :math:`F(\mathbf{v})`
        """
        return -(v.T @ self.b_v) - Σ(softplus(self.W @ v + self.b_h))

    def gibbs_step(self, v0):
        """
        Generate a new visible layer by a gibbs step

        .. math::

            \mathbf{h} \sim P(\mathbf{h}|\mathbf{v})

            \mathbf{v} \sim P(\mathbf{v}|\mathbf{h})

        That means

        .. math::

            \mathbf{h}_{next}  =  P(\mathbf{h}_{next}|\mathbf{v}_0)

            \mathbf{v}_{next}  =  P(\mathbf{v}_{next}|\mathbf{h}_{next})

        :param v0: Visible layer

        :return:
        """
        h0 = self.sample_h_given_v(v0)
        v1 = self.sample_v_given_h(h0)

        return v1

    def sample_h_given_v(self, v):
        """
        With the :math:`P(\mathbf{h} = 1|\mathbf{v})` (obtained from :meth:`.RBM.P_h_given_v`), is generated
        a sample of :math:`\mathbf{h}` with the Bernoulli distribution.

        :param v: Visible layer
        :return: The hidden layer sampled from v
        """
        h_mean = self.P_h_given_v(v)
        h_sample = binomial(n=1, p=h_mean, random_state=None)

        return h_sample

    def P_h_given_v(self, v):
        """
        .. math:: P(\mathbf{h} = 1|\mathbf{v}) = \sigma(\mathbf{v} \mathbf{W}^T + \mathbf{b}^h)

        For :math:`\sigma(x)` see :meth:`~util.util.sigmoid`

        :param v: Visible layer

        :return: :math:`P(\mathbf{h} = 1|\mathbf{v})`.
                 Observe that, as :math:`\mathbf{v}` is a vector, then the return will be a vector of :math:`P(h_i = 1|\mathbf{v})`,
                 for all *i-th* in :math:`\mathbf{h}`.
        """
        return σ(self.W @ v.T + self.b_h)

    def sample_v_given_h(self, h):
        """
        With the :math:`P(\mathbf{v} = 1|\mathbf{h})` (obtained from :meth:`.RBM.P_v_given_h`), is generated
        a sample of :math:`\mathbf{v}` with the Bernoulli distribution.

        :param h: Hidden layer
        :return: The visible layer sampled from h
        """
        v_mean = self.P_v_given_h(h.T)
        v_sample = binomial(n=1, p=v_mean, random_state=None)

        return v_sample

    def P_v_given_h(self, h):
        """
        .. math:: P(\mathbf{v} = 1|\mathbf{h}) = \sigma(\mathbf{h}^T \mathbf{W} + \mathbf{b}^v)

        For :math:`\sigma(x)` see :meth:`~util.util.sigmoid`

        :param h: Hidden layer

        :return: :math:`P(\mathbf{v} = 1|\mathbf{h})`.
                 Observe that, as :math:`\mathbf{h}` is a vector, then the return will be a vector of :math:`P(v_i = 1|\mathbf{h})`,
                 for all *i-th* in :math:`\mathbf{v}`.
        """
        return σ(h.T @ self.W + self.b_v)

    def calculate_parameters_updates(self, v):
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
                        \\frac{1}{N} \sum_{n=1}^{N} \\nabla_{\\theta} f(\mathbf{v}_n)
                    }_\\text{Positive phase}
                  - \\underbrace{
                        \\frac{1}{S} \sum_{n=1}^{S} \\nabla F(\mathbf{\hat{v}}_s)
                    }_\\text{Negative phase}

        where

        * :math:`\mathbf{\hat{v}}_i`: The i-th sample generated

        :param v: `\mathbf{v}` Array visible layer. A mini-batch of :math:`\mathcal{D}`
        :return: The params :math:`\sigma` with the new value
        """
        F = lambda v: self.F(v)
        CD = self.sampling_method
        θ = self.θ
        Ln = self.regularization
        η = self.learning_rate

        # Contrastive divergence
        samples, updates_CD = CD(v)

        # [Expected] negative log-likelihood + Regularization
        cost = mean(F(v)) - mean(F(samples)) + Ln

        # Gradients (use automatic differentiation)
        # We must not compute the gradient through the gibbs sampling, i.e. use consider_constant
        gradients = gradient_descent(cost, wrt=θ, consider_constant=[samples])

        # Updates parameters
        updates = OrderedDict()

        for gradient, parameter in zip(gradients, θ):
            gradient = Gradient(gradient, wrt=parameter)

            updates[parameter] = parameter - η * gradient

        return updates
