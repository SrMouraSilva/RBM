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

from rbm.train.kfold_elements import KFoldElements


def read_data(path, index_col=None):
    if index_col is None:
        index_col = ['index', 'id']

    return pd.read_csv(path, sep=",", index_col=index_col, dtype=np.float32)


def prepare_parameters(rbm_class, i, j, training, validation):
    batch_size = 10

    if rbm_class == RBM:
        learning_rates = [ConstantLearningRate(i) for i in (0.01, 0.05, 0.1)]
    else:
        learning_rates = [ConstantLearningRate(i) for i in (0.05, 0.1, 0.2)]

    return {
        'kfold': [f'{i}/kfold-intern={j}'],
        'data_train': [training],
        'data_validation': [validation],
        'batch_size': [batch_size],
        'hidden_size': [
            #100,
            #500,
            1000
        ],
        'epochs': [batch_size * 100],
        'learning_rate': learning_rates + [
            #Adam(),
            #ADAGRAD(10**-2),
            #AdaMax(),
        ],
        'sampling_method': [
            ContrastiveDivergence(i) for i in (1, )#5
        ] + [
            #PersistentCD(i) for i in (1, )
        ],
        'model_class': [rbm_class],
        'regularization': [
            None
            #L1Regularization(10**-4),
            #L2Regularization(10**-3),
        ],
        'momentum': [
            1 #.9
        ]
    }

# How to execute
# cd experiments && python pedalboards.py
# tensorboard --logdir=experiments/results/logs
original_bag_of_plugins = read_data('data/pedalboard-plugin-full-bag-of-words.csv')

bag_of_plugins = shuffle(original_bag_of_plugins, random_state=42)
kfolds_training_test = KFoldElements(data=bag_of_plugins, n_splits=5, random_state=42, shuffle=False)

for i, original_training, test in kfolds_training_test.split():
    kfolds_training_validation = KFoldElements(data=original_training, n_splits=2, random_state=42, shuffle=False)

    for j, training, validation in kfolds_training_validation.split():
        for rbm_class in [RBM, RBMCF]:
            parameters = prepare_parameters(rbm_class, i, j, training, validation)
            experiment = Experiment()
            experiment.train(parameters)

