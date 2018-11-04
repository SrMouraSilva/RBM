import tensorflow as tf

from rbm.model import Model
from rbm.sampling.contrastive_divergence import ContrastiveDivergence
from rbm.train.persistent import Persistent
from rbm.util.util import Σ, softplus, σ, bernoulli_sample, mean, parameter_name, gradients


class RBM(Model, Persistent):
    """
    :param visible_size: ``D`` Size of the visible layer
    :param hidden_size: ``K`` Size of the hidden layer
    :param SamplingMethod sampling_method: CD or PCD
    """

    def __init__(self, visible_size: int, hidden_size: int, sampling_method=None, **kwargs):
        super(RBM, self).__init__(**kwargs)

        self.visible_size = visible_size
        self.hidden_size = hidden_size

        with tf.name_scope('parameters'):
            self.W = tf.Variable(name='W', initial_value=0.01 * tf.random_normal([self.hidden_size, self.visible_size]),
                                 dtype=tf.float32)
            self.b_h = tf.Variable(name='b_h', dtype=tf.float32, initial_value=tf.zeros([self.hidden_size, 1]))
            self.b_v = tf.Variable(name='b_v', dtype=tf.float32, initial_value=tf.zeros([self.visible_size, 1]))

        self.θ = [self.W, self.b_h, self.b_v]

        self.sampling_method = sampling_method if sampling_method is not None else ContrastiveDivergence()

        self.setup()

    @property
    def parameters(self):
        """
        .. math: \Theta = \{\\boldsymbol{W}, \\boldsymbol{b}^V, \\boldsymbol{b}^h\}

        :return:
        """
        return self.θ

    def setup(self):
        """
        Initialize objects related to the RBM, like the :attr:`~rbm.rbm.RBM.sampling_method`
        and the :attr:`~rbm.rbm.RBM.regularization`
        """
        self.sampling_method.initialize(self)
        self.regularization.initialize(self.W)

    def E(self, v, h):
        """
        Energy function

        .. math::

            E(\\boldsymbol{v}, \\boldsymbol{h}) = - \\boldsymbol{h}^T\\boldsymbol{W}\\boldsymbol{v} - \\boldsymbol{v}^T\\boldsymbol{b}^v - \\boldsymbol{h}^T\\boldsymbol{b}^h

        :param v: :math:`\\boldsymbol{v}` Visible layer
        :param h: :math:`\\boldsymbol{h}` Hidden layer

        :return:
        """
        with tf.name_scope('energy'):
            return h.T @ self.W @ v - (v.T @ self.b_v) - (h.T @ self.b_h)

    def F(self, v):
        """
        The :math:`F(\\boldsymbol{v})` is the free energy function

        .. math::

            F(\\boldsymbol{v}) = - \\boldsymbol{v}^T\\boldsymbol{b}^v - \sum_{i=1}^{K} soft_{+}(\\boldsymbol{W}_{i\cdot} \\boldsymbol{v} + b_i^h)

        Where ``K`` is the :attr:`~rbm.rbm.RBM.hidden_size` (cardinality of the hidden layer)

        For :math:`soft_{+}(x)` see :meth:`~util.util.softplus`

        :param v: :math:`\\boldsymbol{v}` Visible layer
        :return: :math:`F(\\boldsymbol{v})`
        """
        with tf.name_scope('free_energy'):
            return -(v.T @ self.b_v) - Σ(softplus(self.W @ v + self.b_h))

    def gibbs_step(self, v0):
        """
        Generate a new visible layer by a gibbs step

        .. math::

            \\boldsymbol{h} \sim P(\\boldsymbol{h}|\\boldsymbol{v})

            \\boldsymbol{v} \sim P(\\boldsymbol{v}|\\boldsymbol{h})

        That means

        .. math::

            \\boldsymbol{h}^{next}  =  P(\\boldsymbol{h}^{next}|\\boldsymbol{v}^{(0)})

            \\boldsymbol{v}^{next}  =  P(\\boldsymbol{v}^{next}|\\boldsymbol{h}^{next})

        :param v0: :math:`\\boldsymbol{v}^{(0)}` Visible layer

        :return:
        """
        with tf.name_scope('gibbs_step'):
            h0 = self.sample_h_given_v(v0)
            v1 = self.sample_v_given_h(h0)

            return v1

    def sample_h_given_v(self, v):
        """
        With the :math:`P(\\boldsymbol{h} = 1|\\boldsymbol{v})`
        (obtained from :meth:`.RBM.P_h_given_v`), is generated
        a sample of :math:`\\boldsymbol{h}` with the Bernoulli distribution.

        :param v: :math:`\\boldsymbol{v}` Visible layer
        :return: The hidden layer sampled from :math:`\\boldsymbol{v}`
        """
        with tf.name_scope('sample_h_given_v'):
            probabilities = self.P_h_given_v(v)
            h_sample = bernoulli_sample(p=probabilities)

            return h_sample

    def P_h_given_v(self, v):
        """
        .. math:: P(h_i = 1|\\boldsymbol{v}) = \sigma(\\boldsymbol{W}_{i \cdot} \\boldsymbol{v} + b^h)

        .. math:: P(\\boldsymbol{h} = 1|\\boldsymbol{v}) = \\boldsymbol{\sigma}(\\boldsymbol{v} \\boldsymbol{W}^T + \\boldsymbol{b}^h)

        where

        * :math:`\sigma(x)`: Sigmoid (:func:`~rbm.util.util.sigmoid`)
        * :math:`\\boldsymbol{\sigma}(\\boldsymbol{x})`: Return sigmoid vector (sigmiod element-wise)

        :param v: :math:`\\boldsymbol{v}` Visible layer
        :return: :math:`P(\\boldsymbol{v} = 1|\\boldsymbol{h})`.
                 Observe that, as :math:`\\boldsymbol{h}` is a vector, then the return will be a vector of :math:`P(v_i = 1|\\boldsymbol{h})`,
                 for all *i-th* in :math:`\\boldsymbol{v}`.
        """
        with tf.name_scope('P_h_given_v'):
            return σ(self.W @ v + self.b_h)

    def sample_v_given_h(self, h):
        """
        With the :math:`P(\\boldsymbol{v} = 1|\\boldsymbol{h})` (obtained from :meth:`.RBM.P_v_given_h`), is generated
        a sample of :math:`\\boldsymbol{v}` with the Bernoulli distribution.

        :param h: :math:`\\boldsymbol{h}` Hidden layer
        :return: The visible layer :math:`\\boldsymbol{v}` sampled from :math:`\\boldsymbol{h}`
        """
        with tf.name_scope('sample_v_given_h'):
            probability = self.P_v_given_h(h)
            v_sample = bernoulli_sample(p=probability)

            return v_sample

    def P_v_given_h(self, h):
        """
        .. math:: P(v_j=1|\\boldsymbol{h}) = \sigma(\\boldsymbol{h}^T \\boldsymbol{W}_{\cdot j} + b^v_j)

        .. math:: P(\\boldsymbol{v}=1|\\boldsymbol{h}) = \\boldsymbol{\sigma}(\\boldsymbol{h}^T \\boldsymbol{W} + \\boldsymbol{b}^{v^T})^T

        where

        * :math:`\sigma(x)`: Sigmoid (:func:`~rbm.util.util.sigmoid`)
        * :math:`\\boldsymbol{\sigma}(\\boldsymbol{x})`: Return sigmoid vector (sigmoid element-wise)

        :param h: :math:`\\boldsymbol{h}` Hidden layer
        :return: :math:`P(\\boldsymbol{v} = 1|\\boldsymbol{h})`.
                 Observe that, as :math:`\\boldsymbol{h}` is a vector, then the return will be a vector of :math:`P(v_i = 1|\\boldsymbol{h})`,
                 for all *i-th* in :math:`\\boldsymbol{v}`.
        """
        with tf.name_scope('P_v_given_h'):
            return σ(h.T @ self.W + self.b_v.T).T

    def learn(self, v):
        """
        Process for the model learn from a mini-bash from the data :math:`\mathcal{D}`.

        For the learning process, see :class:`.rbm.train.Trainer`.

        For the calc of the :math:`\theta` by the gradients and
        the documentation of the parameters gradients, see
        :meth:`.RBM.calculate_parameters_updates`.

        :param v: :math:`\\boldsymbol{v}` Array visible layer. A mini-batch of :math:`\mathcal{D}`

        :return:
        """
        with tf.name_scope('calculate_parameters'):
            updates = self.calculate_parameters_updates(v)

        assignments = []

        for parameter, update in zip(self.parameters, updates):
            with tf.name_scope(f'assigns/assign_{parameter_name(parameter)}'):
                assignments.append(parameter.assign(update))

        return assignments

    def calculate_parameters_updates(self, v) -> []:
        """
        There are the gradient descent for RBM:

        .. math:: \\nabla_{\\theta} F(\\theta, \mathcal{D}) =
                    \\frac{1}{N} \\underbrace{
                                    \sum_{n=1}^{N} \\nabla_{\\theta} F(\\boldsymbol{v}_n)
                                 }_\\text{Positive phase}
                               - \\underbrace{
                                    \sum_{\\boldsymbol{v}' \in \mathcal{V}} \\nabla_{\\theta} F(\\boldsymbol{v}')
                                 }_\\text{Negative phase}

        where

        * :math:`\mathcal{D}`: A set of N examples. :math:`\mathcal{D} = \{\\boldsymbol{v}_n\}_{n=1}^N`
        * :math:`\mathcal{V}`: All possibilities for the visible layer (:math:`2^D` possibilities).
                 with :math:`D` = size of the visible layer
        * :math:`\\nabla_{\\boldsymbol{W}} F(\\boldsymbol{v})                 \
                    = \mathbb{E}[\\boldsymbol{h}|\\boldsymbol{v}]\\boldsymbol{v}^T \
                    = - \\boldsymbol{\hat{h}}(\\boldsymbol{v})\\boldsymbol{v}^T`
        * :math:`\\nabla_{\\boldsymbol{b}^h} F(\\boldsymbol{v})   \
                    = \mathbb{E}[\\boldsymbol{h}|\\boldsymbol{v}] \
                    = - \\boldsymbol{\hat{h}}(\\boldsymbol{v})`
        * :math:`\\nabla_{\\boldsymbol{b}^v} F(\\boldsymbol{v})   \
                    = - \\boldsymbol{v}`
        * :math:`\\boldsymbol{\hat{h}}(\\boldsymbol{v}) = \\sigma({\\boldsymbol{Wv} + \\boldsymbol{b}^h})`

        But the negative phase are intractable. Then will

        .. note::

            "approximate the expectation under :math:`P(\\boldsymbol{v})`
            with an average of S samples :math:`\mathcal{S} = \{\\boldsymbol{\hat{v}}\}_{s=1}^S`
            draw from :math:`P(\\boldsymbol{v})` i.e. the model."
            -- :cite:`cote2016infinite`

        .. math:: \\nabla_{\\theta} F(\\theta, \mathcal{D}) \\approx
                    \\underbrace{
                        \\frac{1}{N} \sum_{n=1}^{N} \\nabla_{\\theta} F(\\boldsymbol{v}_n)
                    }_\\text{Positive phase}
                  - \\underbrace{
                        \\frac{1}{S} \sum_{n=1}^{S} \\nabla F(\\boldsymbol{\hat{v}}_s)
                    }_\\text{Negative phase}

        where

        * :math:`\\boldsymbol{\hat{v}}_i`: The i-th sample generated

        :param v: `\\boldsymbol{v}` Array visible layer. A mini-batch of :math:`\mathcal{D}`
        :return: TensorFlow variables updated, like the params :math:`\sigma` with the new value
        """
        F = lambda v: self.F(v)
        CD = self.sampling_method
        θ = self.θ
        Ln = self.regularization
        η = self.learning_rate

        # Contrastive divergence
        with tf.name_scope('sampling'):
            samples = CD(v)

        # [Expected] negative log-likelihood + Regularization
        with tf.name_scope('cost'):
            error = mean(F(v)) - mean(F(samples))
            cost = error + Ln

        # Gradients (use automatic differentiation)
        # We must not compute the gradient through the gibbs sampling, i.e. use consider_constant
        grad = gradients(cost, wrt=θ, consider_constant=[samples])

        # Updates parameters
        parameters = []
        for dθ, parameter in grad:
            with tf.name_scope(f'calculate_parameters/calculate_{parameter_name(parameter)}'):
                parameters.append(parameter - η * dθ)

        return parameters

    def __str__(self):
        dicionario = {
            'visible_size': self.visible_size,
            'hidden_size': self.hidden_size,
            'regularization': self.regularization,
            'learning_rate': self.learning_rate,
            'sampling_method': self.sampling_method,
        }

        string = ''
        for k, v in dicionario.items():
            string += f'{k}={v}/'

        return string[:-1]
