from itertools import combinations

import pandas as pd
import tensorflow as tf
from sklearn.utils import shuffle
from tqdm import tqdm

from experiments.data.load_data_util import load_data_one_hot_encoding
from experiments.model_evaluate.evaluate_method.evaluate_method_function import plugins_categories_as_one_hot_encoding
from experiments.rbm_experiment.rbm_experiment import RBMExperiment
from rbm.learning.adam import Adam
from rbm.rbmcf import RBMCF
from rbm.sampling.contrastive_divergence import ContrastiveDivergence
from rbm.train.kfold_cross_validation import KFoldCrossValidation

path = './data/patches-one-hot-encoding.csv'

original_bag_of_plugins = load_data_one_hot_encoding()
bag_of_plugins = shuffle(original_bag_of_plugins, random_state=42)
kfolds_training_test = KFoldCrossValidation(data=bag_of_plugins, n_splits=5, random_state=42, shuffle=False)

rating_size = 117
total_movies = 6
plugins_categories = plugins_categories_as_one_hot_encoding(pd.read_csv("data/plugins_categories_simplified.csv", sep=",", index_col=['id']), rating_size)

metrics = []


def metric(fold, metric: str, operation: dict):
    results = []

    values: dict = session.run(operation)
    keys = values.keys()

    for k, v in values.items():
        results += [{
            'fold': fold,
            'y': k,
            'missing': keys - {k},
            'metric': metric,
            'value': v
        }]

    return results


for i, X_train, X_test in kfolds_training_test.split():
    tf.reset_default_graph()

    with tf.Session() as session:
        hidden_size = 50
        hidden_size = 1000
        hidden_size = 10000
        batch_size, model = 10, RBMCF(total_movies, rating_size, hidden_size=hidden_size, sampling_method=ContrastiveDivergence(1),
                                      learning_rate=Adam(0.05), momentum=0)

        model.load(session, f"./results/model/kfold={i}+kfold-intern=0+batch_size={batch_size}+{model.__str__().replace('/', '+')}/rbm.ckpt")

        model = RBMExperiment(model, total_movies)

        for j in tqdm(range(1, 6)):
            for y_columns in combinations(range(6), j):
                metrics += metric(i, 'Accuracy', model.accuracy(X_test.values, y_columns=y_columns))
                metrics += metric(i, 'Hit@5',    model.hit_ratio(X_test.values, y_columns=y_columns, k=5, n_labels=rating_size))
                metrics += metric(i, 'MDCG',     model.mdcg(X_test.values, y_columns=y_columns, n_labels=rating_size))
                metrics += metric(i, 'MAP@5',    model.map(X_test.values, y_columns=y_columns, k=5, n_labels=rating_size, plugins_categories_as_one_hot_encoding=plugins_categories))
                ##metrics['MRR'].append(session.run(model.mrr(X_test.values, y_column=j)))


frame = pd.DataFrame(metrics)
#frame.to_csv('rbm_results-early.csv')

print(frame.head(5))
#for k, v in metrics.items():
#    print(k.ljust(10), np.mean(v), np.std(v))
