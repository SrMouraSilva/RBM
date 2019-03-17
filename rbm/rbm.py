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

    def __init__(self, visible_size: int, hidden_size: int, sampling_method=None, momentum=1, b_v=None, **kwargs):
        super().__init__(**kwargs)

        self.visible_size: int = visible_size
        self.hidden_size: int = hidden_size

        if b_v is None:
            b_v = tf.zeros([self.visible_size, 1])

        with tf.name_scope('parameters'):
            self.W = tf.Variable(name='W', initial_value=0.01 * tf.random_normal([self.hidden_size, self.visible_size]),
                                 dtype=tf.float32)
            self.b_h = tf.Variable(name='b_h', dtype=tf.float32, initial_value=tf.zeros([self.hidden_size, 1]))
            self.b_v = tf.Variable(name='b_v', dtype=tf.float32, initial_value=b_v)

            self.ΔW = tf.Variable(name='dW', initial_value=tf.zeros([self.hidden_size, self.visible_size]), dtype=tf.float32)
            self.Δb_v = tf.Variable(name='db_v', initial_value=tf.zeros([self.visible_size, 1]), dtype=tf.float32)
            self.Δb_h = tf.Variable(name='db_h', initial_value=tf.zeros([self.hidden_size, 1]), dtype=tf.float32)

        self.sampling_method = sampling_method if sampling_method is not None else ContrastiveDivergence()
        self.momentum = momentum

        self.setup()

    @property
    def θ(self):
        return [self.W, self.b_h, self.b_v]

    @property
    def parameters(self):
        """
        .. math: \Theta = \{\\boldsymbol{W}, \\boldsymbol{b}^V, \\boldsymbol{b}^h\}

        :return:
        """
        return super().parameters

    def setup(self):
        """
        Initialize objects related to the RBM, like the :attr:`~rbm.rbm.RBM.sampling_method`
        and the :attr:`~rbm.rbm.RBM.regularization`
        """
        self.sampling_method.initialize(self)
        self.regularization.initialize(self)

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
            return - (h.T @ self.W @ v) - (v.T @ self.b_v) - (h.T @ self.b_h)

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
            return -(v.T @ self.b_v) - Σ(softplus(self.W @ v + self.b_h), axis=0).to_vector()

    def marginalize_over_v(self, h):
        with tf.name_scope('marginalize_over_v'):
            return -(h.T @ self.b_h) - Σ(softplus(h.T @ self.W + self.b_v), axis=0).to_vector()

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

    '''
    def learn(self, *args, **kwargs) -> []:
        """
        Process for the model learn from a mini-bash from the data :math:`\mathcal{D}`.

        For the learning process, see :class:`.rbm.train.Trainer`.

        For the calc of the :math:`\theta` by the gradients and
        the documentation of the parameters gradients, see
        :meth:`.RBM.calculate_parameters_updates`.

        :param v: :math:`\\boldsymbol{v}` Array visible layer. A mini-batch of :math:`\mathcal{D}`

        :return:
        """
        return super().learn(*args, **kwargs)
    '''

    def calculate_parameters_updates(self, v, *args, **kwargs) -> []:
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
        momentum = self.momentum

        # Contrastive divergence
        with tf.name_scope('sampling'):
            samples = CD(v)

        # [Expected] negative log-likelihood + Regularization
        with tf.name_scope('cost'):
            error = mean(F(v), axis=0) - mean(F(samples), axis=0)
            cost = error + Ln

        # Gradients (use automatic differentiation)
        # We must not compute the gradient through the gibbs sampling, i.e. use consider_constant
        grad = gradients(cost, wrt=θ, consider_constant=[samples])

        # Updates parameters
        parameters = []
        for dθ, parameter in grad:
            with tf.name_scope(f'calculate_parameters/calculate_{parameter_name(parameter)}'):
                parameters.append(momentum * parameter - η * dθ)

        return parameters

    def pseudo_likelihood(self):
        #https://github.com/ethancaballero/Restricted_Boltzmann_Machine__RBM/blob/master/rbm.py#L152-L162
        #https://github.com/monsta-hd/boltzmann-machines/blob/master/boltzmann_machines/rbm/base_rbm.py#L496-L513
        pass

    def learn(self, v0, *args, **kwargs):
        '''
        Based on "On the Model Selection of Bernoulli Restricted Boltzmann Machines Through Harmony Search"
         - https://www.researchgate.net/profile/Gustavo_De_Rosa/publication/287772009_On_the_Model_Selection_of_Bernoulli_Restricted_Boltzmann_Machines_Through_Harmony_Search/links/5799503108aec89db7bb9c48/On-the-Model-Selection-of-Bernoulli-Restricted-Boltzmann-Machines-Through-Harmony-Search.pdf
         - https://github.com/monsta-hd/boltzmann-machines/blob/master/boltzmann_machines/rbm/base_rbm.py
        '''
        η = self.learning_rate
        α = self.momentum
        λ = self.regularization

        with tf.name_scope('gibbs_chain'):
            P_h0_given_v0 = self.P_h_given_v(v0)
            h0 = bernoulli_sample(p=P_h0_given_v0)

            P_v1_given_h0 = self.P_v_given_h(h0)
            v1 = bernoulli_sample(p=P_v1_given_h0)

            P_h1_given_v1 = self.P_h_given_v(v1)
            h1 = bernoulli_sample(p=P_h1_given_v1)

        batch_size = tf.shape(v0)[1].cast(tf.float32)

        with tf.name_scope('delta_W'):
            ΔW = η * (P_h0_given_v0 @ v0.T - P_h1_given_v1 @ v1.T) / batch_size - η*(λ*self.W) + α*self.ΔW
            self.ΔW = self.ΔW.assign(ΔW)

        with tf.name_scope('delta_v_b'):
            Δb_v = η * mean(v0 - v1, axis=1).to_vector() + α*self.Δb_v
            self.Δb_v = self.Δb_v.assign(Δb_v)

        with tf.name_scope('delta_h_b'):
            Δb_h = η * mean(P_h0_given_v0 - P_h1_given_v1, axis=1).to_vector() + α*self.Δb_h
            self.Δb_h = self.Δb_h.assign(Δb_h)

        with tf.name_scope(f'assigns/params'):
            W_update = self.W.assign(self.W + self.ΔW)
            b_v_update = self.b_v.assign(self.b_v + self.Δb_v)
            b_h_update = self.b_h.assign(self.b_h + self.Δb_h)

        return [W_update, b_h_update, b_v_update]

    def __str__(self):
        dicionario = {
            'class': self.__class__.__name__,
            'visible_size': self.visible_size,
            'hidden_size': self.hidden_size,
            'regularization': self.regularization,
            'learning_rate': self.learning_rate,
            'sampling_method': self.sampling_method,
            'momentum': self.momentum,
        }

        string = ''
        for k, v in dicionario.items():
            string += f'{k}={v}/'

        return string[:-1]
