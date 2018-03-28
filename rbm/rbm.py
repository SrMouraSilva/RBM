import numpy as np

from rbm.model import Model
from util.util import σ, softplus, Σ, mean


class RBM(Model):
    """
    Based in https://github.com/MarcCote/iRBM/blob/master/iRBM/models/rbm.py

    :param input_size: ``D`` Size of the visible layer
    :param hidden_size: ``K`` Size of the hidden layer
    :param SamplingMethod sampling_method: CD or PCD
    """

    def __init__(self, input_size, hidden_size, sampling_method, *args, **kwargs):
        super(RBM, self).__init__(*args, **kwargs)

        self.input_size = input_size
        self.hidden_size = hidden_size

        # TODO - Talvez as dimenções estejam trocadas
        self.W = np.zeros((self.hidden_size, self.input_size), dtype=np.float64)
        self.b_h = np.zeros(self.hidden_size, dtype=np.float64)
        self.b_v = np.zeros(self.input_size, dtype=np.float64)

        self.θ = [self.W, self.b_h, self.b_v]

        self.sampling_method = sampling_method

        self.setup()

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
        The ``F(v)`` is the free energy function

        .. math::

            F(\mathbf{v}) = - \mathbf{v}^T\mathbf{b}^v - \sum_{i=1}^{K} soft_{+}(\mathbf{W}_{i\cdot} \mathbf{v} + b_i^h)

        Where ``K`` is the :attr:`~rbm.rbm.RBM.hidden_size` (cardinality of the hidden layer)

        For

        .. math:: soft_{+}(x)

        see :meth:`~util.util.softplus`

        :param v: Visible layer
        :return: ``F(v)``
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

    def sample_h_given_v(self, v, return_probs=False):
        """
        .. math:: P(\mathbf{h}|\mathbf{v}) = \sigma(\mathbf{v} \mathbf{W}^T + \mathbf{b}_h)

        For

        .. math:: \sigma(x)

        see :meth:`~util.util.sigmoid`

        :param v: Visible layer
        :param return_probs: ??

        :return: The hidden sampled from v
        """
        h_mean = σ(self.W @ v.T + self.b_h)
        return h_mean

        if return_probs:
            return h_mean

        h_sample = self.theano_rng.binomial(size=h_mean.shape, n=1, p=h_mean, dtype=theano.config.floatX)
        return h_sample

    def sample_v_given_h(self, h, return_probs=False):
        """
        .. math:: P(\mathbf{v}|\mathbf{h}) = \sigma(\mathbf{h} \mathbf{W} + \mathbf{b}_v)

        For

        .. math:: \sigma(x)

        see :meth:`~util.util.sigmoid`

        :param h: Hidden layer
        :param return_probs: ??

        :return: The visible layer sampled from h
        """
        v_mean = σ(h.T @ self.W + self.b_v)
        return v_mean

        if return_probs:
            return v_mean

        v_sample = self.theano_rng.binomial(size=v_mean.shape, n=1, p=v_mean, dtype=theano.config.floatX)
        return v_sample

    def get_updates(self, v):
        """
        There are the gradient descent for RBM:

        .. math:: \\nabla_{\\theta} F(\\theta, \mathcal{D}) =
                    \\frac{1}{N} \sum_{n=1}^{N} \\nabla_{\\theta} F(\mathbf{v}_n)
                               - \sum_{\mathbf{v}' \in \mathcal{V}} \\nabla_{\\theta} F(\mathbf{v}')

        where

        * :math:`\mathcal{D}`: A set of N examples. :math:`\mathcal{D} = \{\mathbf{v}_n\}_{n=1}^N`
        * :math:`\\nabla_{\mathbf{W}} F(\mathbf{v})                 \
                    = \mathbb{E}[\mathbf{h}|\mathbf{v}]\mathbf{v}^T \
                    = - \mathbf{\^h}(\mathbf{v})\mathbf{v}^T`
        * :math:`\\nabla_{\mathbf{b}^h} F(\mathbf{v})   \
                    = \mathbb{E}[\mathbf{h}|\mathbf{v}] \
                    = - \mathbf{\^h}(\mathbf{v})`
        * :math:`\\nabla_{\mathbf{b}^v} F(\mathbf{v})   \
                    = - \mathbf{v}`
        * :math:`\mathbf{\^h}(\mathbf{v}) = \\sigma({\mathbf{Wv} + \mathbf{b}^h})`

        But the negative phase (:math:`\sum_{\mathbf{v}' \in \mathcal{V}} \\nabla_{\\theta} F(\mathbf{v}')`)
        are intractable. Then will

        .. note::

            "approximate the expectation under :math:`P(\mathbf{v})`
            with an average of S samples :math:`\mathcal{S} = \{\mathbf{\^v}\}_{s=1}^S`
            draw from :math:`P(\mathbf{v})` i.e. the model."
            -- Infinite RBM

        .. math:: \\nabla_{\\theta} F(\\theta, \mathcal{D}) \\approx
                    \\frac{1}{N} \sum_{n=1}^{N} \\nabla_{\\theta} f(\mathbf{v}_n)
                  - \\frac{1}{S} \sum_{n=1}^{S} \\nabla F(\mathbf{\^v}_s)

        :param v: Array visible layer. A mini-batch of :math:`\mathcal{D}`
        :return: The params :math:`\sigma` with the new value
        """
        F = lambda v: self.F(v)
        λ = lambda g: self.λ(g)
        CD = self.sampling_method
        θ = self.θ
        Ln = self.regularization

        # Contrastive divergence
        samples, updates_CD = CD(v)

        # [Expected] negative log-likelihood
        cost = mean(F(v)) - mean(F(samples)) + Ln

        # Gradients (use automatic differentiation)
        # We must not compute the gradient through the gibbs sampling, i.e. use consider_constant
        y = cost
        x = θ

        gparams = T.grad(y, x, consider_constant=[samples])
        gradients = dict(zip(θ, gparams))

        # ESSA PARTE TODA ABAIXO É PARA ATUALIZAR OS PARÂMETROS
        # Get learning rates for all params given their gradient.
        lr, updates_lr = λ(gradients)

        updates = OrderedDict()
        updates.update(updates_CD)  # Add updates from CD
        updates.update(updates_lr)  # Add updates from learning_rate

        # Updates parameters
        for param, gparam in gradients.items():
            updates[param] = param - lr[param] * gradients[param]

        return updates
