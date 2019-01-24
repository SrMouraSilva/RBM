from abc import ABCMeta

import pandas as pd
import tensorflow as tf

from experiments.other_models.other_model import OtherModel
from rbm.util.rank import mean_reciprocal_rank


class EvaluateMethod(metaclass=ABCMeta):
    def evaluate(self, model: OtherModel, x, y, label=None) -> float:
        pass


class Accuracy(EvaluateMethod):
    def evaluate(self, model: OtherModel, x, y, label=None):
        y_generated = model.predict(x)
        return self.accuracy(y, y_generated)

    def accuracy(self, y: pd.DataFrame, y_generated):
        count = sum(y.values == y_generated)

        return count / len(y_generated)


class MRR(EvaluateMethod):
    """
    Mean Reciprocal Rank
    """
    def evaluate(self, model: OtherModel, x, y, label=None):
        recommendations = model.recommends(x)
        return self.mrr(y, recommendations)

    def mrr(self, y, recommendations):
        session = tf.get_default_session()
        session_created = False

        if session is None:
            session_created = True
            session = tf.Session()

        depth = len(recommendations[0])

        y = tf.one_hot(y, depth=depth).eval(session=session)
        score = mean_reciprocal_rank(y, recommendations).eval(session=session)

        if session_created:
            session.close()

        return score
