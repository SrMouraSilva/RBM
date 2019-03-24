from collections import defaultdict

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils import shuffle

from experiments.model_evaluate.evaluate_method.evaluate_method import plugins_categories_as_one_hot_encoding
from experiments.rbm_experiment.rbm_experiment import RBMExperiment
from rbm.learning.adam import Adam
from rbm.rbmcf import RBMCF
from rbm.train.kfold_elements import KFoldElements

path = './data/patches-one-hot-encoding.csv'

original_bag_of_plugins = pd.read_csv(path, sep=",", index_col=['index', 'id'], dtype=np.float32)
bag_of_plugins = shuffle(original_bag_of_plugins, random_state=42)
kfolds_training_test = KFoldElements(data=bag_of_plugins, n_splits=5, random_state=42, shuffle=False)

rating_size = 117
total_movies = 6
plugins_categories = plugins_categories_as_one_hot_encoding(pd.read_csv("data/plugins_categories_simplified.csv", sep=",", index_col=['id']), rating_size)

for i, X_trian, X_test in kfolds_training_test.split():
    with tf.Session() as session:

        model = RBMCF(total_movies, rating_size, hidden_size=500, learning_rate=Adam(0.05), momentum=0)
        model.load(session, f"./results/model/kfold={i}+kfold-intern=0+batch_size={64}+{model.__str__().replace('/', '+')}/rbm.ckpt")

        model = RBMExperiment(model, total_movies)

        metrics = defaultdict(list)

        for i in range(6):
            metrics['accuracy'].append(session.run(model.accuracy(X_test.values, y_column=i)))
            metrics['hit@5'].append(session.run(model.hit_ratio(X_test.values, y_column=i, k=5, n_labels=rating_size)))
            metrics['mrr'].append(session.run(model.mrr(X_test.values, y_column=i)))
            metrics['mdcg'].append(session.run(model.mdcg(X_test.values, y_column=i, n_labels=rating_size)))
            metrics['map@5'].append(session.run(model.map(X_test.values, y_column=i, k=5, n_labels=rating_size, plugins_categories_as_one_hot_encoding=plugins_categories)))

        for k, v in metrics.items():
            print(k.ljust(10), np.mean(v))

    break
