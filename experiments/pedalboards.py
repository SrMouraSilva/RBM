import tensorflow as tf
from sklearn.utils import shuffle

from experiments.data.load_data_util import load_data_one_hot_encoding
from experiments.experiment import Experiment
from rbm.learning.adam import Adam
from rbm.learning.adamax import AdaMax
from rbm.learning.adaptive_learning_rate import AdaptiveLearningRate
from rbm.learning.tf_decay_learning_rate import TFDecayLearningRate
from rbm.rbmcf import RBMCF
from rbm.learning.adagrad import ADAGRAD
from rbm.learning.constant_learning_rate import ConstantLearningRate
from rbm.rbm import RBM
from rbm.regularization.regularization import NoRegularization, L1AutoGradRegularization, L2AutoGradRegularization, \
    L2
from rbm.sampling.contrastive_divergence import ContrastiveDivergence
from rbm.sampling.persistence_contrastive_divergence import PersistentCD
from sklearn.model_selection import KFold

from rbm.train.defined_decay import DefinedDecay
from rbm.train.kfold_cross_validation import KFoldCrossValidation



def prepare_parameters(rbm_class, i, j, training, validation):
    batch_size = 10
    #batch_size = 32
    #batch_size = 64

    epochs = batch_size * 150

    if rbm_class == RBM:
        learning_rates = [ConstantLearningRate(i) for i in (0.005, 0.01, 0.05, 0.1, 0.2)]
        learning_rates = []
    else:
        learning_rates = [ConstantLearningRate(i) for i in (0.005, 0.01, 0.05, 0.1, 0.2)]
        learning_rates = []

    return {
        'kfold': [f'{i}/kfold-intern={j}'],
        'data_train': [training],
        'data_validation': [validation],
        'batch_size': [batch_size],
        'hidden_size': [
            50,
            #500,
            #1000,
            #10000,
        ],
        'epochs': [epochs],
        'learning_rate': learning_rates + [
            #TFDecayLearningRate(lambda epoch: tf.train.cosine_decay(0.05, epoch, epochs + 1)),
            #TFDecayLearningRate(lambda epoch: tf.train.cosine_decay_restarts(0.05, epoch, 10)),
            #TFDecayLearningRate(lambda epoch: tf.train.exponential_decay(0.05, epoch, epochs+1, 0.005)),
            #TFDecayLearningRate(lambda epoch: tf.train.inverse_time_decay(0.05, epoch, epochs+1, 0.005)),
            #TFDecayLearningRate(lambda epoch: tf.train.linear_cosine_decay(0.05, epoch, epochs+1)),
            #TFDecayLearningRate(lambda epoch: tf.train.natural_exp_decay(0.05, epoch, epochs + 1, 0.005)),
            #TFDecayLearningRate(lambda epoch: tf.train.noisy_linear_cosine_decay(0.05, epoch, epochs + 1)),
            #TFDecayLearningRate(lambda epoch: tf.train.polynomial_decay(0.05, epoch, epochs + 1)),

            #AdaptiveLearningRate(lambda: tf.linspace(0.05, 1e-4, num=epochs+1)),
            #Adam(alpha=AdaptiveLearningRate(lambda: tf.linspace(0.05, 1e-4, num=epochs+1))),
            #Adam(alpha=TFLearningRate(lambda epoch: tf.train.exponential_decay(0.05, epoch, epochs + 1, 0.005))),

            #Adam(0.02),
            Adam(0.05),
            #Adam(0.001),
            #Adam(0.0001),
            #ADAGRAD(10**-2),
            #AdaMax(),
        ],
        'sampling_method': [
            ContrastiveDivergence(1),
            ContrastiveDivergence(5),
            ContrastiveDivergence(
                DefinedDecay(lambda: tf.concat(axis=0, values=[
                    tf.ones(shape=(200, )) * 1,
                    tf.ones(shape=(300, )) * 3,
                    tf.ones(shape=(500, )) * 5,
                    tf.ones(shape=(500+1, )) * 7,
                ]))
            )
        ] + [
            #PersistentCD(i) for i in (1, )
        ],
        'model_class': [rbm_class],
        'regularization': [
            None,
            #L2(1e-5),
            #L1Regularization(10**-4),
            #L2Regularization(10**-4),
        ],
        'momentum': [
            0,
            #.5,
        ],
        #'persist': [True]
        'persist': [False],
        'log_epoch_step': [10]
    }

# How to execute
# cd experiments && python pedalboards.py
# tensorboard --logdir=experiments/results/logs
original_bag_of_plugins = load_data_one_hot_encoding()

bag_of_plugins = shuffle(original_bag_of_plugins, random_state=42)
kfolds_training_test = KFoldCrossValidation(data=bag_of_plugins, n_splits=5, random_state=42, shuffle=False)

for i, original_training, test in kfolds_training_test.split():
    kfolds_training_validation = KFoldCrossValidation(data=original_training, n_splits=2, random_state=42, shuffle=False)

    # Train + Validation (not Test)
    #for j, training, validation in kfolds_training_validation.split():
    #    for rbm_class in [RBM, RBMCF]:
    #        parameters = prepare_parameters(rbm_class, i, j, training, validation)
    #        experiment = Experiment()
    #        experiment.train(parameters)

    # Train + Test
    #for rbm_class in [RBM, RBMCF]:
    for rbm_class in [RBMCF]:
        parameters = prepare_parameters(rbm_class, i, 0, original_training, test)
        experiment = Experiment()
        experiment.train(parameters)

    break
