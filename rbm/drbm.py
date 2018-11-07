from rbm.rbm import RBM
import tensorflow as tf

from rbm.util.util import softmax, σ, Σ, 𝚷, bernoulli_sample, gradients, mean, parameter_name


class DRBM(RBM):
    """
    Discriminative RBM
    http://www.dmi.usherb.ca/~larocheh/publications/icml-2008-discriminative-rbm.pdf

    :param visible_size: ``D`` Size of the visible layer (``x``)
    :param hidden_size: ``K`` Size of the hidden layer (``h``)
    :param target_class_size: ``C`` Size of the class layer (``y``)
    """

    def __init__(self, visible_size: int, hidden_size: int, target_class_size: int, **kwargs):
        super(DRBM, self).__init__(visible_size, hidden_size, **kwargs)

        self.target_class_size = target_class_size

        with tf.name_scope('parameters'):
            self.U = tf.Variable(name='U', initial_value=0.01 * tf.random_normal([self.hidden_size, self.target_class_size]),
                                 dtype=tf.float32)
            self.b_y = tf.Variable(name='b_y', dtype=tf.float32, initial_value=tf.zeros([self.target_class_size, 1]))

        self.θ += [self.b_y, self.U]

        self.sampling_method = None

    def E(self, y, v, h) -> tf.Tensor:
        with tf.name_scope('energy'):
            return - (h.T @ self.W @ v) - (v.T @ self.b_v) - (h.T @ self.b_h) \
                   - (y.T @ self.b_y) - (h.T @ self.U @ y)

    def F(self, v):
        raise Exception('Not implemented')

    def sample_h_given_v(self, v):
        raise Exception('Not implemented')

    def P_h_given_v(self, v):
        raise Exception('Not implemented')

    def gibbs_step(self, v0, y0):
        v0, v1, y0, y1, probabilities_h0, probabilities_h1 = self.full_gibbs_step(v0, y0)
        return v1, y1

    def full_gibbs_step(self, v0, y0):
        with tf.name_scope('contrastive_divergence'):
            with tf.name_scope('positive_phase'):
                probabilities_h0 = self.P_h_given_y_v(y0, v0)

            sample_h_given_y_v = lambda y, v: bernoulli_sample(p=probabilities_h0)

            with tf.name_scope('negative_phase'):
                h0 = sample_h_given_y_v(y0, v0)
                y1 = self.sample_y_given_h(h0)
                v1 = self.sample_v_given_h(h0)

                probabilities_h1 = self.P_h_given_y_v(y1, v1)

        return v0, v1, y0, y1, probabilities_h0, probabilities_h1

    def sample_y_given_h(self, h):
        with tf.name_scope('sample_y_given_h'):
            probabilities = self.P_y_giver_h(h)
            y_sample = bernoulli_sample(p=probabilities)

            return y_sample

    def P_y_giver_h(self, h):
        with tf.name_scope('P_y_given_h'):
            return softmax(self.U.T @ h + self.b_y)

    def P_h_given_y_v(self, y, v):
        with tf.name_scope('P_h_given_v_y'):
            return σ(self.W @ v + self.b_h + self.U @ y)

    def P_y_given_v(self, category, v):
        """
        P(y=category | v)
        """
        exp = tf.exp
        b_y = self.b_y

        C = self.target_class_size
        K = self.hidden_size

        with tf.name_scope('P_y_given_v'):
            Wv = tf.reshape(self.W @ v, (v.shape[1], -1, 1))
            eq = self.b_h + self.U + Wv

            numerator = exp(b_y[category]) * 𝚷(eq[:, :, category], axis=1)
            denominator = Σ([exp(b_y[y]) * 𝚷(eq[:, :, y], axis=1) for y in range(C)])

            return numerator / denominator

    def calculate_parameters_updates(self, v, y=None) -> []:
        E = self.E
        θ = self.θ
        Ln = self.regularization
        η = self.learning_rate

        v0, v1, y0, y1, probabilities_h0, probabilities_h1 = self.full_gibbs_step(v, y)

        # [Expected] negative log-likelihood + Regularization
        with tf.name_scope('cost'):
            E0 = E(y0, v0, probabilities_h0)
            E1 = E(y1, v1, probabilities_h1)

            error = mean(E0 - E1)
            cost = error + Ln

        # Gradients (use automatic differentiation)
        # We must not compute the gradient through the gibbs sampling, i.e. use consider_constant
        grad = gradients(cost, wrt=θ, consider_constant=[v0, v1, probabilities_h0, probabilities_h1, y0, y1])

        # Updates parameters
        parameters = []
        for dθ, parameter in grad:
            with tf.name_scope(f'calculate_parameters/calculate_{parameter_name(parameter)}'):
                parameters.append(parameter - η * dθ)

        return parameters
