import time
from collections import OrderedDict
from itertools import product
from typing import Iterable, Dict

import pandas as pd
import tensorflow as tf

from rbm.cfrbm import CFRBM
from rbm.learning.constant_learning_rate import ConstantLearningRate
from rbm.rbm import RBM
from rbm.sampling.contrastive_divergence import ContrastiveDivergence
from rbm.sampling.persistence_contrastive_divergence import PersistentCD
from rbm.train.task.persistent_task import PersistentTask
from rbm.train.task.rbm_inspect_histograms_task import RBMInspectHistogramsTask
from rbm.train.task.rbm_inspect_scalars_task import RBMInspectScalarsTask
from rbm.train.task.summary_task import SummaryTask
from rbm.train.trainer import Trainer


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


def train(data, batch_size=10, epochs=100, hidden_size=100, learning_rate=None, regularization=None,
          sampling_method=None, persist=False, model_class=None):
    """
    # Batch_size = 10 or 100
    # https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf
    """

    tf.set_random_seed(42)

    total_elements, size_element = data.shape

    if model_class is None or model_class == RBM:
        rbm = RBM(
            visible_size=size_element,
            hidden_size=hidden_size,
            regularization=regularization,
            learning_rate=learning_rate,
            sampling_method=sampling_method,
        )
    else:
        rbm = CFRBM(
            movies_size=6,
            ratings_size=size_element/6,
            hidden_size=hidden_size,
            regularization=regularization,
            learning_rate=learning_rate,
            sampling_method=sampling_method,
        )

    trainer = Trainer(rbm, data, batch_size=batch_size)

    trainer.stopping_criteria.append(lambda current_epoch: current_epoch > epochs)

    log = f"results/logs/{rbm}/{time.time()}"

    trainer.tasks.append(RBMInspectScalarsTask())
    trainer.tasks.append(RBMInspectHistogramsTask())
    trainer.tasks.append(SummaryTask(log=log))
    #trainer.tasks.append(BeholderTask(log='results/logs'))

    if persist:
        trainer.tasks.append(PersistentTask(path=f"results/model/batch_size={batch_size}/{rbm}/rbm.ckpt"))

    trainer.train()


class Experiment:

    def train(self, cross_validation):
        parameters = self.prepare_parameters(cross_validation)

        for kwargs in parameters:
            tf.reset_default_graph()
            train(**kwargs)

    def prepare_parameters(self, cross_validation) -> Iterable[Dict]:
        cross_validation = OrderedDict(cross_validation)

        create_kwargs = lambda x: {k: v for k, v in zip(cross_validation.keys(), x)}

        return map(create_kwargs, product(*cross_validation.values()))


# jupyter notebook notebooks/
# tensorboard --logdir=experiments/results/logs
# cd experiments && python pedalboards.py

# Treinar
#bag_of_plugins = read_data('data/pedalboard-plugin-bag-of-words.csv')
#bag_of_plugins = treat_input(bag_of_plugins)

bag_of_plugins = read_data('data/pedalboard-plugin-full-bag-of-words.csv')

#bag_of_plugins = read_data('data/clash-royale-bag-of-words.csv', index_col=['index'])

cross_validation = {
    'data': [bag_of_plugins],
    'batch_size': [10],
    'hidden_size': [100, 1000],
    'epochs': [300],
    'learning_rate': [
        ConstantLearningRate(i) for i in (10**-3, 10**-2, 5 * 10**-2, 10**-1, 5 * 10**-1, 1)
    ],
    'sampling_method': [
        ContrastiveDivergence(i) for i in (1,)#(1, 5)
    ] + [
        #PersistentCD(i, shape=(117, 10)) for i in (1, 5)
    ],
    'model_class': [
        RBM, CFRBM
    ]
}

experiment = Experiment()
experiment.train(cross_validation)
