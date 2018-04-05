from collections import OrderedDict

from rbm.learning.learning_rate import LearningRate


class ADAGRAD(LearningRate):
    def __init__(self, lr, eps=1e-6):
        """
        http://ruder.io/optimizing-gradient-descent/index.html#adagrad

        Implements the ADAGRAD learning rule.

        Parameters
        ----------
        lr: float
            learning rate
        eps: float
            eps needed to avoid division by zero.

        Reference
        ---------
        Duchi, J., Hazan, E., & Singer, Y. (2010).
        Adaptive subgradient methods for online learning and stochastic optimization.
        Journal of Machine Learning
        """
        super(ADAGRAD, self).__init__(lr)

        self._updates = OrderedDict()
        self.epsilon = eps
        self.parameters = []

    @property
    def updates(self):
        return self._updates

    def __mul__(self, other):
        """
        :param ~util.Gradient other:
        :return:
        """
        return self.calculate()

    def __rmul__(self, other):
        return other

    def __call__(self, grads):
        # O que eu vou retornar
        learning_rates = OrderedDict()

        params_names = map(lambda p: p.name, self.parameters)
        for param in grads.keys():
            # sum_squared_grad := \sum g_t^2
            sum_squared_grad = sharedX(param.get_value() * 0.)

            if param.name is not None:
                sum_squared_grad.name = 'sum_squared_grad_' + param.name

            # Check if param is already there before adding
            if sum_squared_grad.name not in params_names:
                self.parameters.append(sum_squared_grad)
            else:
                sum_squared_grad = self.parameters[params_names.index(sum_squared_grad.name)]

            # Accumulate gradient
            new_sum_squared_grad = sum_squared_grad + T.sqr(grads[param])

            # Compute update
            root_sum_squared = T.sqrt(new_sum_squared_grad + self.epsilon)

            # Apply update
            self.updates[sum_squared_grad] = new_sum_squared_grad
            learning_rates[param] = self.learning_rate / root_sum_squared

        return learning_rates

    def get_lr(self, param):
        params_names = map(lambda p: p.name, self.parameters)
        idx_param = params_names.index('sum_squared_grad_' + param.name)
        sum_squared_grad = self.parameters[idx_param]
        root_sum_squared = np.sqrt(sum_squared_grad.get_value() + self.epsilon)
        lr = self.base_lr / root_sum_squared
        return lr
