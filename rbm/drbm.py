from rbm.rbm import RBM
import tensorflow as tf

from rbm.util.util import softmax, σ, Σ, 𝚷, bernoulli_sample, gradients, mean, parameter_name


class DRBM(RBM):
    """
    http://www.dmi.usherb.ca/~larocheh/publications/icml-2008-discriminative-rbm.pdf

    :param visible_size: ``D`` Size of the visible layer (``x``)
    :param hidden_size: ``K`` Size of the hidden layer (``h``)
    :param target_class_size: ``C`` Size of the class layer (``y``)
    """

    def __init__(self, visible_size: int, hidden_size: int, target_class_size: int, **kwargs):
        super().__init__(visible_size, hidden_size, **kwargs)

        self.target_class_size = target_class_size

        with tf.name_scope('parameters'):
            self.U = tf.Variable(name='U', initial_value=0.01 * tf.random_normal([self.hidden_size, self.target_class_size]),
                                 dtype=tf.float32)
            self.b_y = tf.Variable(name='b_y', dtype=tf.float32, initial_value=tf.zeros([self.target_class_size, 1]))

        self.θ += [self.b_y, self.U]

    def E(self, y, v, h) -> tf.Tensor:
        with tf.name_scope('energy'):
            return - (h.T @ self.W @ v) - (v.T @ self.b_v) - (h.T @ self.b_h) \
                   - (y.T @ self.b_y) - (h.T @ self.U @ y)

    def F(self, v):
        raise Exception('Not implemented')

    def gibbs_step(self, v0):
        raise Exception('Not implemented')

    def sample_h_given_v(self, v):
        raise Exception('Not implemented')

    def P_h_given_v(self, v):
        raise Exception('Not implemented')

    def sample_y_given_h(self, h):
        with tf.name_scope('sample_y_given_h'):
            probabilities = self.P_y_giver_h(h)
            y_sample = bernoulli_sample(p=probabilities)

            return y_sample

    def P_y_giver_h(self, h):
        with tf.name_scope('P_y_given_h'):
            return softmax(h.T @ self.W + self.b_y.T)

    def P_h_given_v_y(self, v, y):
        with tf.name_scope('P_h_given_v_y'):
            # y-th column of U
            Uy = self.U[:, y]

            return σ(self.W @ v + self.b_h + Uy)

    def P_y_given_v(self, category, v):
        """
        P(y=category | v)
        """
        exp = tf.exp
        b_y = self.b_y

        C = self.target_class_size
        K = self.hidden_size

        U = self.U
        W = self.W

        with tf.name_scope('P_y_given_v'):
            f = lambda y: [1 + exp(b_y[j] + U[j, y] + W[j] @ v) for j in range(K)]

            numerator = exp(b_y) * 𝚷(f(category))
            denominator = Σ([exp(b_y) * 𝚷(f(y)) for y in range(C)])

            return numerator / denominator

    def learn(self, y, v):
        with tf.name_scope('calculate_parameters'):
            updates = self.calculate_parameters_updates(y, v)

        assignments = []

        for parameter, update in zip(self.parameters, updates):
            with tf.name_scope(f'assigns/assign_{parameter_name(parameter)}'):
                assignments.append(parameter.assign(update))

    def calculate_parameters_updates(self, y, v) -> []:
        E = self.E
        θ = self.θ
        Ln = self.regularization
        η = self.learning_rate

        # Contrastive divergence
        with tf.name_scope('positive_phase'):
            y_0 = y
            v_0 = v
            probabilities_h_0 = self.P_y_given_v(y_0, v_0)

        sample_h_given_y_v = lambda y, v: bernoulli_sample(p=probabilities_h_0)

        with tf.name_scope('negative_phase'):
            h_0 = sample_h_given_y_v(y_0, v_0)
            y_1 = self.sample_y_given_h(h_0)
            v_1 = self.sample_v_given_h(h_0)
            probabilities_h_1 = self.P_y_given_v(y_1, v_1)

        # [Expected] negative log-likelihood + Regularization
        with tf.name_scope('cost'):
            E0 = E(y_0, v_0, probabilities_h_0)
            E1 = E(y_1, v_1, probabilities_h_1)

            error = mean(E0 - E1)
            cost = error + Ln

        # Gradients (use automatic differentiation)
        # We must not compute the gradient through the gibbs sampling, i.e. use consider_constant
        grad = gradients(cost, wrt=θ, consider_constant=[v_0, v_1, probabilities_h_0, probabilities_h_1, y_0, y_1])

        # Updates parameters
        parameters = []
        for dθ, parameter in grad:
            with tf.name_scope(f'calculate_parameters/calculate_{parameter_name(parameter)}'):
                parameters.append(parameter - η * dθ)

        return parameters
