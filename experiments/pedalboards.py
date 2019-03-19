import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.utils import shuffle

from experiments.experiment import Experiment
from rbm.learning.adam import Adam
from rbm.learning.adamax import AdaMax
from rbm.learning.adaptive_learning_rate import AdaptiveLearningRate
from rbm.learning.tf_learning_rate import TFLearningRate
from rbm.rbmcf import RBMCF
from rbm.learning.adagrad import ADAGRAD
from rbm.learning.constant_learning_rate import ConstantLearningRate
from rbm.rbm import RBM
from rbm.regularization.regularization import NoRegularization, L1Regularization, L2Regularization, \
    ConstantRegularization
from rbm.sampling.contrastive_divergence import ContrastiveDivergence
from rbm.sampling.persistence_contrastive_divergence import PersistentCD
from sklearn.model_selection import KFold

from rbm.train.kfold_elements import KFoldElements


def read_data(path, index_col=None):
    if index_col is None:
        index_col = ['index', 'id']

    return pd.read_csv(path, sep=",", index_col=index_col, dtype=np.float32)


def prepare_parameters(rbm_class, i, j, training, validation):
    #batch_size = 32
    batch_size = 64
    epochs = batch_size * 100
    #epoch = batch_size * 150

    if rbm_class == RBM:
        learning_rates = [ConstantLearningRate(i) for i in (0.005, 0.01, 0.05, 0.1, 0.2)]
    else:
        learning_rates = [ConstantLearningRate(i) for i in (0.005, 0.01, 0.05, 0.1, 0.2)]
        learning_rates = []

    return {
        'kfold': [f'{i}/kfold-intern={j}'],
        'data_train': [training],
        'data_validation': [validation],
        'batch_size': [batch_size],
        'hidden_size': [
            500,
            #100,
            #250, 500, 1000,
            #1, 2, 5, 10, 25, 50, 100,
            #50, 100, 200
            #500, 1000
        ],
        'epochs': [epochs],
        'learning_rate': learning_rates + [
            #TFLearningRate(lambda epoch: tf.train.cosine_decay(0.05, epoch, epochs + 1)),
            #TFLearningRate(lambda epoch: tf.train.cosine_decay_restarts(0.05, epoch, 10)),
            #TFLearningRate(lambda epoch: tf.train.exponential_decay(0.05, epoch, epochs+1, 0.005)),
            #TFLearningRate(lambda epoch: tf.train.inverse_time_decay(0.05, epoch, epochs+1, 0.005)),
            #TFLearningRate(lambda epoch: tf.train.linear_cosine_decay(0.05, epoch, epochs+1)),
            #TFLearningRate(lambda epoch: tf.train.natural_exp_decay(0.05, epoch, epochs + 1, 0.005)),
            #TFLearningRate(lambda epoch: tf.train.noisy_linear_cosine_decay(0.05, epoch, epochs + 1)),
            #TFLearningRate(lambda epoch: tf.train.polynomial_decay(0.05, epoch, epochs + 1)),

            AdaptiveLearningRate(lambda: tf.linspace(0.05, 1e-4, num=epochs+1)),
            Adam(alpha=AdaptiveLearningRate(lambda: tf.linspace(0.05, 1e-4, num=epochs+1))),
            Adam(alpha=TFLearningRate(lambda epoch: tf.train.exponential_decay(0.05, epoch, epochs + 1, 0.005))),
            Adam(0.05),
            Adam(0.001),
            Adam(0.0001),
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
            #ConstantRegularization(1e-5),
            None,
            #L1Regularization(10**-4),
            #L2Regularization(10**-4),
        ],
        'momentum': [
            # New train task
            0,
            #.5,
        ],
        #'persist': [True]
        'persist': [False]
    }

# How to execute
# cd experiments && python pedalboards.py
# tensorboard --logdir=experiments/results/logs
#original_bag_of_plugins = read_data('data/pedalboard-plugin-full-bag-of-words.csv')
original_bag_of_plugins = read_data('data/patches-one-hot-encoding.csv')

bag_of_plugins = shuffle(original_bag_of_plugins, random_state=42)
kfolds_training_test = KFoldElements(data=bag_of_plugins, n_splits=5, random_state=42, shuffle=False)

for i, original_training, test in kfolds_training_test.split():
    kfolds_training_validation = KFoldElements(data=original_training, n_splits=2, random_state=42, shuffle=False)

    # Train + Validation (not Test)
    #for j, training, validation in kfolds_training_validation.split():
    #    for rbm_class in [RBM, RBMCF]:
    #        parameters = prepare_parameters(rbm_class, i, j, training, validation)
    #        experiment = Experiment()
    #        experiment.train(parameters)

    # Train + Test
    #for rbm_class in [RBM, RBMCF]:
    for rbm_class in [RBMCF, RBM]:
        parameters = prepare_parameters(rbm_class, i, 0, original_training, test)
        experiment = Experiment()
        experiment.train(parameters)

    break
