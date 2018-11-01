import pandas as pd

from rbm.cfrbm import ExpectationSamplingMethod, TopKProbabilityElementsMethod, NotSampleMethod, RBMLikeMethod
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


def train(dataset, batch_size=10, epochs=100, persist=False, visible_sampling_method=None):
    """
    # Batch_size = 10 or 100
    # https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf
    """
    import time

    import tensorflow as tf

    from rbm.learning.constant_learning_rate import ConstantLearningRate
    from rbm.rbm import RBM
    from rbm.sampling.persistence_contrastive_divergence import PersistentCD
    from rbm.cfrbm import CFRBM
    from rbm.regularization.regularization import L1Regularization, L2Regularization
    from rbm.train.task.beholder_task import BeholderTask
    from rbm.train.task.persistent_task import PersistentTask
    from rbm.train.task.rbm_mensurate_task import RBMMeasureTask
    from rbm.train.task.summary_task import SummaryTask

    from rbm.train.trainer import Trainer

    tf.set_random_seed(42)

    total_elements, size_element = dataset.shape

    rbm = RBM(
        #movies_size=1,
        #ratings_size=size_element,
        visible_size=size_element,
        hidden_size=100,
        #visible_sampling_method=visible_sampling_method,
        #regularization=L1Regularization(0.01),
        #regularization=L2Regularization(0.01),
        learning_rate=ConstantLearningRate(10**-2),
        #sampling_method=PersistentCD(25),
    )

    trainer = Trainer(rbm, dataset, batch_size=batch_size)
    trainer.stopping_criteria.append(lambda current_epoch: current_epoch > epochs)

    log = "results/logs/{}/{}/{}".format(visible_sampling_method.__class__.__name__, batch_size, time.time())

    trainer.tasks.append(RBMMeasureTask())
    trainer.tasks.append(SummaryTask(log=log))
    trainer.tasks.append(BeholderTask(log='results/logs'))

    if persist:
        trainer.tasks.append(PersistentTask(path="results/model/{}/rbm.ckpt".format(batch_size)))

    trainer.train()


# jupyter notebook notebooks/
# tensorboard --logdir=experiments/results/logs
# cd experiments && python pedalboards.py
# Treinar
bag_of_plugins = read_data('data/pedalboard-plugin-bag-of-words.csv')
bag_of_plugins = treat_input(bag_of_plugins)

#bag_of_plugins = read_data('data/clash-royale-bag-of-words.csv', index_col=['index'])

#train(bag_of_plugins, batch_size=100, epochs=200)
#train(bag_of_plugins, batch_size=100, epochs=50, visible_sampling_method=ExpectationSamplingMethod())
#train(bag_of_plugins, batch_size=100, epochs=50, visible_sampling_method=TopKProbabilityElementsMethod(8))
#train(bag_of_plugins, batch_size=10, epochs=500, visible_sampling_method=NotSampleMethod())
train(bag_of_plugins, batch_size=10, epochs=100, visible_sampling_method=RBMLikeMethod())
