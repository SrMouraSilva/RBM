import numpy as np
from sklearn.utils import check_X_y

__author__ = "Saulo Oliveira <saulo.freitas.oliveira@gmail.com>"
__status__ = "production"
__version__ = "1.0.0"
__date__ = "07 September 2018"


def roundint(n, v):
    return int(round(n)) if np.isscalar(n) else v


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


class PrototypeSelection():

    def select(self, X, y=None):
        X, y = check_X_y(X, y, multi_output=True, y_numeric=True)
        return np.ravel(np.ones(len(X), dtype=bool)), X, y


class RandomSelection(PrototypeSelection):

    def __init__(self, k=None):
        self.k = k

    def select(self, X, y=None):
        n = len(X)

        k = kheuristic(self.k, X)

        perm = np.random.permutation(n)
        perm = perm[:k]

        idx = np.zeros(n, dtype=bool)

        idx[perm] = True

        idx = np.ravel(idx)

        return idx, X[idx], y[idx]


class RegEnnSelection(PrototypeSelection):

    def __init__(self, k=None, a=1):
        self.k = k
        self.a = a
        self.knn = None

    def select(self, X, y=None):
        from sklearn.neighbors import KNeighborsRegressor

        k = kheuristic(self.k, X)
        selection = np.ravel(np.ones((1, len(X)), dtype=bool))
        self.knn = KNeighborsRegressor(n_neighbors=k)

        for i in range(len(X)):
            self.knn.fit(X[selection])
            _, nn = self.knn.kneighbors(X[i])

            theta = self.a * np.std(X[nn])

            selection[i] = False

            self.knn.fit(X[selection])
            if abs(y[i] - self.knn.predict(X[i])) <= theta:
                selection[i] = True

        selection = np.ravel(selection)

        return selection, X[selection], y[selection]


class ActiveSelection():
    '''
    Optimized Fixed-Size Kernel Models for Large Data Sets
    '''

    def __init__(self, subset_size=None, lam=0, trials=None):
        self.lam = lam
        self.subset_size = subset_size
        self.trials = trials

    def select(self, X, y):

        n = len(X)
        k = round(np.asscalar(np.log10(n) * 5))

        self.subset_size = roundint(self.subset_size, k)
        self.trials = roundint(self.trials, 100)
        self.lam = roundint(self.lam, 0)

        perm = np.random.permutation(n)
        perm = perm[:k]

        Xr, yr = X[perm], y[perm]

        old_crit = self.__quad_renyi_entropy(Xr, self.lam)

        for trial in range(self.trials):
            old_Xr = Xr
            old_yr = yr

            i = np.random.randint(0, self.subset_size)
            j = np.random.randint(0, len(X))

            Xr[i] = X[j]
            yr[i] = y[j]

            crit = self.__quad_renyi_entropy(Xr, self.lam)

            if crit > old_crit:
                # undo permutation
                Xr = old_Xr
                yr = old_yr
            else:
                old_crit = crit

        idx = np.unique(np.where(np.isin(X, Xr).all(axis=1)))
        return idx, X[idx], y[idx]

    @staticmethod
    def __quad_renyi_entropy(X, l):

        from scipy.spatial.distance import cdist
        from scipy.linalg import eig

        Dx = cdist(X, X)

        di = np.diag_indices(len(X))

        Dx[di] += l

        lam, U = eig(Dx)

        # en = -log((sum(U, 1) / n). ^ 2 * lam);

        mu = np.ravel(np.sum(np.asmatrix(U), axis=0) / len(X))

        return -np.log(np.real((mu ** 2).T @ lam))


class NLSelection():

    def __init__(self, distance=1):
        self.distance = distance

    @staticmethod
    def __order_of(X):
        from scipy.spatial.distance import cdist

        x_origin = np.min(X, axis=0)
        keys = np.ravel(cdist(np.asmatrix(x_origin), X))

        return np.argsort(keys, axis=0)

    def select(self, X, y):
        from scipy.signal import find_peaks
        order = __class__.__order_of(X)

        if self.distance is None or self.distance == 0:
            from sklearn.neighbors import NearestNeighbors
            knn = NearestNeighbors(n_neighbors=2).fit(X, y)
            dist, idx = knn.kneighbors(X)
            self.distance = max(round(np.std(y[idx[:, 1]])), 1)

        yl = np.ravel(y[order])

        s = 1

        h_peaks, _ = find_peaks(+yl, prominence=(None, +s * np.std(yl)))
        l_peaks, _ = find_peaks(-yl, prominence=(-s * np.std(yl), None))

        peaks = np.hstack((h_peaks, l_peaks))

        idx = np.ravel(np.zeros((1, len(X)), dtype=bool))

        idx[peaks] = True

        return idx, X[idx], y[idx]


class KSSelection():

    def __init__(self, high_cutoff=0.2, low_cutoff=0.32, k=None):
        self.cutoff = (high_cutoff, low_cutoff)
        self.k = k

    @staticmethod
    def __pval_ks_2smap(entry):
        from scipy.stats import ks_2samp, zscore

        a = zscore(entry[0]) if np.std(entry[0]) > 0 else entry[0]

        b = zscore(entry[1]) if np.std(entry[1]) > 0 else entry[1]

        _, pval = ks_2samp(a, b)

        return pval

    def select(self, X, y):
        from sklearn.neighbors import NearestNeighbors

        n = len(X)

        self.k = kheuristic(self.k, n)

        knn = NearestNeighbors(n_neighbors=int(self.k + 1), algorithm='ball_tree').fit(X)

        distx, ind = knn.kneighbors(X)

        knn = NearestNeighbors(n_neighbors=int(self.k + 1), algorithm='ball_tree').fit(y)

        disty, ind = knn.kneighbors(y, return_distance=True)

        zipped = list(zip(distx[:, :1], disty[:, :1]))

        p = [self.__pval_ks_2smap(entry) for entry in zipped]

        order = np.argsort(p, axis=0)

        h_cutoff = round(self.cutoff[0] * n)
        l_cutoff = round(self.cutoff[1] * n)

        idx = np.zeros((n, 1), dtype=bool).ravel()
        idx[order[:l_cutoff]] = True
        idx[order[-h_cutoff:]] = True

        return idx, X[idx], y[idx]

    @staticmethod
    def __otsu(pval):
        info = np.ravel(pval).reshape(1, -1)

        his, bins = np.histogram(info, np.array(range(0, 100)))

        w = 1 / (info.shape[0] * info.shape[1])

        var_fcn = lambda t: (np.sum(his[:t]) * w) * (np.sum(his[t:]) * w) * (
                np.mean(his[:t]) - np.mean(his[t:])) ** 2

        values = [var_fcn(t) for t in bins[1:-1]]

        return np.asscalar(np.max(np.ravel(values)))


class DROP2_RE():

    def __init__(self, k=None, a=0.1):
        from sklearn.neighbors import KNeighborsRegressor
        super(DROP2_RE, self).__init__()
        self.k = k
        self.alpha = a
        self.model = KNeighborsRegressor()

    def __err(self, Xy, associates, i):
        X, y = Xy

        self.model.fit(X[associates], y[associates])

        error_with = np.abs(self.model.predict(np.asmatrix(X[i])) - y[i])

        associates_without_x = list(set(associates) - set([i]))

        self.model.fit(X[associates_without_x], y[associates_without_x])

        error_without = np.abs(self.model.predict(np.asmatrix(X[i])) - y[i])

        return np.asarray([error_with, error_without])

    def __theta(self, y, A):

        associates = A[:self.k] if len(A) >= self.k else A

        return self.alpha * np.std(y[associates])

    def select(self, X, y):
        from sklearn.neighbors import NearestNeighbors
        _, associates = NearestNeighbors(n_neighbors=len(X)).fit(X).kneighbors(X)

        Xy = (X, y)

        selected = np.ones(len(X), dtype=bool)

        for i in range(len(X)):
            A = associates[i]

            errors = np.zeros(2)

            for a in A:
                err = np.array(self.__err(Xy, A, a)).ravel()
                t = self.__theta(y, associates[a])
                errors += np.asarray(err < t)

            # error with <= error without
            if errors[0] <= errors[1]:
                selected[i] = False

                for a in A:
                    associates[a] = np.delete(associates[a], np.where(associates[a] == i))

        return selected, X[selected], y[selected]


class MutualInformationSelection():

    def __init__(self, k=None, alpha=0.05):
        self.k = k
        self.alpha = alpha

    def select(self, X, y):
        from sklearn.feature_selection import mutual_info_regression
        from sklearn.preprocessing import MinMaxScaler
        from sklearn.neighbors import NearestNeighbors
        from dit.shannon import mutual_information

        n = len(X)

        if self.k is None:
            self.k = np.round(np.log(n))

        mask = np.arange(n)

        mi = [mutual_information((X[mask != i], y[mask != i])) for i in range(n)]

        mi = MinMaxScaler().fit_transform(mi)

        _, neighbors = NearestNeighbors(n_neighbors=self.k + 1).fit(X).kneighbors(X)

        # dropout themselves
        neighbors = neighbors[:, 1:]

        mask_as_set = set(mask)

        not_neighbors = [list(mask_as_set - set(neighbors[i])) for i in range(n)]

        # mutual information without neighbors
        nn_mi = [mutual_info_regression(X[not_neighbors[i]], y[not_neighbors[i]]) for i in range(n)]

        selected = np.zeros(len(X), dtype=bool)

        for i in range(n):
            cdiff = np.asarray([(mi[i] - mi[k]) for k in neighbors[i]])

            selected[i] = np.sum(cdiff > self.alpha) < self.k

        return selected, X[selected], y[selected]
