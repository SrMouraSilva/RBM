import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from experiments.experiment import Experiment
from rbm.learning.adam import Adam
from rbm.learning.adamax import AdaMax
from rbm.rbmcf import RBMCF
from rbm.learning.adagrad import ADAGRAD
from rbm.learning.constant_learning_rate import ConstantLearningRate
from rbm.rbm import RBM
from rbm.regularization.regularization import NoRegularization, L1Regularization, L2Regularization
from rbm.sampling.contrastive_divergence import ContrastiveDivergence
from rbm.sampling.persistence_contrastive_divergence import PersistentCD
from sklearn.model_selection import KFold


def read_data(path, index_col=None):
    if index_col is None:
        index_col = ['index', 'id']

    return pd.read_csv(path, sep=",", index_col=index_col, dtype=np.float32)


# How to execute
# tensorboard --logdir=experiments/results/logs
# cd experiments && python pedalboards.py
original_bag_of_plugins = read_data('data/pedalboard-plugin-full-bag-of-words.csv')


bag_of_plugins = shuffle(original_bag_of_plugins, random_state=42)
kfolds = KFold(n_splits=10, random_state=42, shuffle=False)

for kfold, (train_index, test_index) in enumerate(kfolds.split(bag_of_plugins)):
    train = bag_of_plugins.iloc[train_index]
    test = bag_of_plugins.iloc[test_index]

    batch_size = 10

    cross_validation = {
        'kfold': [kfold],
        'data_train': [train],
        'data_validation': [test],
        'batch_size': [batch_size],
        #'hidden_size': [500, 1000],
        'hidden_size': [100],
        'epochs': [batch_size * 100],
        'learning_rate': [
            ConstantLearningRate(i) for i in (0.01, 0.05, 0.1, 0.20)
        ] + [
            #Adam(),
            #ADAGRAD(10**-2),
            #AdaMax(),
        ],
        'sampling_method': [
            ContrastiveDivergence(i) for i in (1, )#5
        ] + [
            #PersistentCD(i) for i in (1, )
        ],
        'model_class': [
            RBMCF, RBM
        ],
        'regularization': [
            None
            #L1Regularization(10**-4),
            #L2Regularization(10**-3),
        ],
        'momentum': [
            1 #.9
        ]
    }

    experiment = Experiment()
    experiment.train(cross_validation)
