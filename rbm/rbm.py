import numpy as np

from rbm.model import Model
from util.util import sigmoid


class RBM(Model):
    """
    Based in https://github.com/MarcCote/iRBM/blob/master/iRBM/models/rbm.py
    """

    def __init__(self, input_size, hidden_size, *args, **kwargs):
        """
        :param input_size: ``D`` Size of the visible layer
        :param hidden_size: ``K`` Size of the hidden layer
        """
        super(RBM, self).__init__(*args, **kwargs)

        self.input_size = input_size
        self.hidden_size = hidden_size

        # TODO - Talvez as dimenções estejam trocadas
        self.W = np.zeros((self.hidden_size, self.input_size), dtype=np.float64)
        self.b_h = np.zeros(self.hidden_size, dtype=np.float64)
        self.b_v = np.zeros(self.input_size, dtype=np.float64)

        self.parameters = [self.W, self.b_h, self.b_v]

        self.setup()

    def setup(self):
        """
        Initialize the weight (``W``)
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

    def gibbs_step(self, visible_layer):
        """
        Generate a new visible layer by a gibbs step

        .. math::

            \mathbf{h} \sim P(\mathbf{h}|\mathbf{v})

            \mathbf{v} \sim P(\mathbf{v}|\mathbf{h})

        That means

        .. math::

            \mathbf{h}_{next}  =  P(\mathbf{h}_{next}|\mathbf{visible\_layer})

            \mathbf{v}_{next}  =  P(\mathbf{v}_{next}|\mathbf{h}_{next})

        :param visible_layer: Visible layer

        :return:
        """
        h0 = self.sample_h_given_v(visible_layer)
        v1 = self.sample_v_given_h(h0)

        return v1

    def sample_h_given_v(self, v, return_probs=False):
        """
        .. math:: P(\mathbf{h}|\mathbf{v}) = \sigma(\mathbf{v} \mathbf{W}^T + \mathbf{b}_h)

        Where sigmoid is

        .. math:: \sigma(x) = 1/(1 + e^{-x})

        :param v: Visible layer
        :param return_probs: ??

        :return: The hidden sampled from v
        """
        h_mean = sigmoid(self.W @ v.T + self.b_h)
        return h_mean

        if return_probs:
            return h_mean

        h_sample = self.theano_rng.binomial(size=h_mean.shape, n=1, p=h_mean, dtype=theano.config.floatX)
        return h_sample

    def sample_v_given_h(self, h, return_probs=False):
        """
        .. math:: P(\mathbf{v}|\mathbf{h}) = \sigma(\mathbf{h} \mathbf{W} + \mathbf{b}_v)

        Where sigmoid is

        .. math:: \sigma(x) = 1/(1 + e^{-x})

        :param h: Hidden layer
        :param return_probs: ??

        :return: The visible layer sampled from h
        """
        v_mean = sigmoid(h.T @ self.W + self.b_v)
        return v_mean

        if return_probs:
            return v_mean

        v_sample = self.theano_rng.binomial(size=v_mean.shape, n=1, p=v_mean, dtype=theano.config.floatX)
        return v_sample
