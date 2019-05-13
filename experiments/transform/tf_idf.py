import numpy as np
from sklearn.base import TransformerMixin

from rbm.util.embedding import one_hot_encoding, bag_of_words


class VectorTFIDFTransform(TransformerMixin):
    def __init__(self, n_labels):
        self.number_of_times_term_t_appears_in_a_document = None
        self.n_labels = n_labels

    def fit(self, X, *_):
        """
        Expected X as numbers
        """
        X = bag_of_words(X, depth=self.n_labels)

        self.number_of_times_term_t_appears_in_a_document = X.sum(axis=0)

        total_of_documents = X.shape[0]
        number_of_documents_with_term = (X > 0).sum(axis=0)

        self.idf = -np.log(number_of_documents_with_term / total_of_documents)

        return self

    def transform(self, X, *_):
        X_bag = bag_of_words(X, depth=self.n_labels)
        frequency_for_each_document = X_bag / X_bag.sum(axis=1).reshape((-1, 1))

        tf = np.log(frequency_for_each_document)

        res = tf * self.idf
        res[np.isinf(res)] = 0

        return res
