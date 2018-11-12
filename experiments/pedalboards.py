import pandas as pd
from sklearn.model_selection import train_test_split

from experiments.experiment import Experiment
from rbm.cfrbm import CFRBM
from rbm.learning.constant_learning_rate import ConstantLearningRate
from rbm.rbm import RBM
from rbm.sampling.contrastive_divergence import ContrastiveDivergence


def read_data(path, index_col=None):
    if index_col is None:
        index_col = ['index', 'id']

    return pd.read_csv(path, sep=",", index_col=index_col)


def treat_input(bag_of_plugins):
    """
    Consider only pedalboards with more then 3 distinct plugins
    """
    # Convert only zero and one
    bag_of_plugins = ((bag_of_plugins > 0) * 1)

    # Remove guitar_patches.id column
    #del bag_of_plugins['id']
    # Remove never unused plugin
    # > bag_of_plugins.T[bag_of_plugins.sum() == 0]
    # 9
    #del bag_of_plugins['9']

    # Zero all that will disconsider
    # BOMB(9) and None(107)
    bag_of_plugins['9'] = 0
    bag_of_plugins['107'] = 0

    # Consider only pedalboards with more then 3 distinct plugins
    bag_of_plugins = bag_of_plugins[bag_of_plugins.T.sum() > 3]
    return bag_of_plugins


# jupyter notebook notebooks/
# tensorboard --logdir=experiments/results/logs
# cd experiments && python pedalboards.py

# Treinar
#bag_of_plugins = read_data('data/pedalboard-plugin-bag-of-words.csv')
#bag_of_plugins = treat_input(bag_of_plugins)

bag_of_plugins = read_data('data/pedalboard-plugin-full-bag-of-words.csv')
x_train, x_test = train_test_split(bag_of_plugins, test_size=.2, random_state=42)

#bag_of_plugins = read_data('data/clash-royale-bag-of-words.csv', index_col=['index'])

cross_validation = {
    'data': [bag_of_plugins],
    'data_x': [x_train],
    'data_y': [None],
    'batch_size': [10],
    'hidden_size': [10, 50, 100, 500, 1000, 5000],
    'epochs': [300],
    'learning_rate': [
        ConstantLearningRate(i) for i in (10**-3, 10**-2, 5 * 10**-2, 10**-1, 5 * 10**-1, 1)
    ],
    'sampling_method': [
        ContrastiveDivergence(i) for i in (1, 5)
    ] + [
        #PersistentCD(i, shape=(117, 10)) for i in (1, 5)
    ],
    'model_class': [
        CFRBM#, RBM
    ]
}


'''
cross_validation = {
    'data_x': [x_train],
    'data_y': [None],
    'batch_size': [10],
    'hidden_size': [50],
    'epochs': [300],
    'learning_rate': [
        ConstantLearningRate(i) for i in (10**-1, )
    ],
    'sampling_method': [
        ContrastiveDivergence(i) for i in (1, )
    ] + [
        #PersistentCD(i, shape=(117, 10)) for i in (1, 5)
    ],
    'model_class': [
        CFRBM
    ]
}
'''

experiment = Experiment()
experiment.train(cross_validation)
