from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from scipy.optimize import root
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.validation import check_X_y
from experiments.other_models.mlm.protosel import RandomSelection

from numpy.linalg import matrix_power

__all__ = ['MinimalLearningMachine',
           'MinimalLearningMachineClassifier',
           'NearestNeighborMinimalLearningMachineClassifier',
           'CubicMinimalLearningMachine']

__author__ = "Saulo Oliveira <saulo.freitas.oliveira@gmail.com>"
__status__ = "production"
__version__ = "1.0.0"
__date__ = "07 September 2018"


def r2(a, b):
    from scipy.stats import pearsonr
    p, _ = pearsonr(np.ravel(a), np.ravel(b))
    return np.asscalar(p ** 2)


def kheuristic(k, n):
    try:
        iterator = iter(n)
        n = len(n)
    except TypeError:
        n = None

    if np.isinf(k) or k is None:
        k = n
    elif np.isscalar(k):
        if 0 < k <= 1:
            k = round(np.asscalar(k * n))
        else:
            k = round(np.asscalar(max(min(k, n), 3)))
    else:
        k = round(np.asscalar(np.log10(n) * 5))

    if n is None:
        raise Exception('The number of elements in the set can not be none')


class RandomSelection():

    def __init__(self, k=None):
        self.k = k

    def select(self, X, y=None):
        n = len(X)

        k = kheuristic(self.k, X)

        perm = np.random.permutation(n)
        perm = perm[:k]

        idx = np.ravel(np.zeros((1, n), dtype=bool))

        idx[perm] = True

        idx = np.ravel(idx)

        return idx, X[idx], y[idx]


class MinimalLearningMachine(BaseEstimator, RegressorMixin):
    _estimator_type = 'regressor'

    def __init__(self, selector=None, bias=False, l=0):
        self.selector = RandomSelection(k=np.inf) if selector is None else selector
        self.M = []
        self.t = []
        self._sparsity_scores = (0, np.inf, 0)  # sparsity and norm frobenius
        self.bias = bias
        self.regularization = l

    def fit(self, X, y):
        from numpy.linalg import norm

        X, y = check_X_y(X, y, multi_output=True, y_numeric=True)

        idx, _, _ = self.selector.select(X, y)

        if y.ndim == 1:
            y = np.asmatrix(y).T

        X = np.asmatrix(X)

        self.M = X[idx, :]
        self.t = y[idx, :]

        assert (len(self.M) != 0), "No reference point was yielded by the selector"

        dx = cdist(X, self.M)
        dy = cdist(y, self.t)

        if self.bias:
            dx = np.hstack((np.ones((len(X), 1)), dx))

        if np.isscalar(self.regularization) and self.regularization > 0:
            dx[idx, :] = dx[idx, :] + self.regularization

        inv_dx = np.linalg.pinv(dx)

        self.B_ = inv_dx @ dy

        fb_norm = norm(self.B_, ord='fro')
        n_fb_norm = fb_norm / (norm(inv_dx, ord='fro') * norm(dy, ord='fro'))

        self._sparsity_scores = (1 - len(self.M) / len(X), fb_norm, n_fb_norm)

        return self

    def mulat_(self, y, dyh):

        dy2t = cdist(np.asmatrix(self.t), np.asmatrix(y))

        dyh = np.ravel(dyh)
        dy2t = np.ravel(dy2t)

        return np.sum((dyh ** 2 - dy2t ** 2) ** 2)

    def active_(self, dyhat):
        y0h = np.mean(self.t, axis=0)

        result = [root(method='lm', fun=lambda y: self.mulat_(y, dyh), x0=y0h) for dyh in dyhat]
        yhat = list(map(lambda y: y.x, result))
        return np.asarray(yhat)

    def predict(self, X, y=None):
        try:
            getattr(self, "B_")
        except AttributeError:
            raise RuntimeError("You must train classifier before predicting data!")

        X = np.asmatrix(X)

        dx = cdist(X, self.M)

        if self.bias:
            dx = np.hstack((np.ones((len(X), 1)), dx))

        dyhat = dx @ self.B_

        return self.active_(dyhat)

    def sparsity(self):
        try:
            getattr(self, "B_")
        except AttributeError:
            raise RuntimeError("You must train classifier before predicting data!")

        s = np.round(self._sparsity_scores, 2)

        return tuple(s)

    def score(self, X, y, sample_weight=None):
        return r2(y, self.predict(X))


class MinimalLearningMachineClassifier(MinimalLearningMachine, ClassifierMixin):

    def __init__(self, selector=None, bias=False, l=0):
        MinimalLearningMachine.__init__(self, selector, bias, l)
        self._estimator_type = 'classifier'
        self.lb = LabelBinarizer()

    def fit(self, X, y=None):
        self.lb.fit(y)
        return MinimalLearningMachine.fit(self, X, self.lb.transform(y))

    def active_(self, dyhat):
        classes = self.lb.transform(self.lb.classes_)

        result = [np.argmin(list(map(lambda y_class: self.mulat_(y_class, dyh), classes))) for dyh in dyhat]

        return self.lb.inverse_transform(classes[result])

    def score(self, X, y, sample_weight=None):
        return ClassifierMixin.score(self, X, y, sample_weight)


class NearestNeighborMinimalLearningMachineClassifier(MinimalLearningMachineClassifier):

    def __init__(self, selector=None):
        MinimalLearningMachineClassifier.__init__(self, selector)

    def active_(self, dyhat):
        m = np.argmin(dyhat, 1)
        return self.t[m]


class CubicMinimalLearningMachine(MinimalLearningMachine):

    def active_(self, dyhat):
        a = len(self.t)
        b = -3 * np.sum(self.t)
        c = 3 * np.sum(np.power(self.t, 2)) - np.sum(np.power(dyhat, 2), axis=1)
        d = np.power(dyhat, 2) @ self.t - np.sum(np.power(self.t, 3))

        return [self.cases_(np.roots([a, b, c[i], d[i]]), dyhat[i]) for i in range(len(dyhat))]

    def cases_(self, roots, dyhat):
        r = list(map(np.isreal, roots))
        if np.sum(r) == 3:
            # Rescue the root with the lowest cost associated
            j = [self.mulat_(y, dyhat) for y in np.real(roots)]
            return np.real(roots[np.argmin(j)])
        else:
            # As True > False, then rescue the first real value
            return np.real(roots[np.argmax(r)])
