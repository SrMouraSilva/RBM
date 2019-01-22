from abc import ABCMeta

import pandas as pd
import tensorflow as tf
from rbm.util.rank import mean_reciprocal_rank


class EvaluateMethod(metaclass=ABCMeta):
    def evaluate(self, model, x, y, label=None) -> float:
        pass


class Accuracy(EvaluateMethod):
    def evaluate(self, model, x, y, label=None):
        y_generated = model.recommends(x)
        return self.accuracy(y, y_generated)

    def accuracy(self, y: pd.DataFrame, y_generated):
        count = sum(y.values == y_generated)

        return count / len(y_generated)


class MRR(EvaluateMethod):
    def evaluate(self, model, x, y, label=None):
        model.recommends(x)

    def mrr(self, y, recommendations, model):
        y = tf.one_hot(y, depth=117).eval(session=model._session)
        return mean_reciprocal_rank(y, recommendations).eval(session=model._session)
