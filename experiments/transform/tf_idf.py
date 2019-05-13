import numpy as np
from sklearn.base import TransformerMixin


class VectorTFIDFTransform(TransformerMixin):
    def __init__(self):
        self.tf = None

    def fit(self, X, *_):
        """
        Expected X as bag of words
        """
        self.tf = X.sum(axis=0) / X.sum()

        return self

    def transform(self, X, *_):
        total_of_documents = X.shape[0]
        number_of_documents_with_term = (X > 0).sum(axis=0)

        return np.log(total_of_documents / number_of_documents_with_term)
