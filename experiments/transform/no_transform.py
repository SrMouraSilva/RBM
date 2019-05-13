from sklearn.base import TransformerMixin


class NoTransform(TransformerMixin):

    def __init__(self):
        self.tf = None

    def fit(self, X, *_):
        return self

    def transform(self, X, *_):
        return X
