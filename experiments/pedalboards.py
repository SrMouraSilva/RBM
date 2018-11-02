import time

import pandas as pd
import tensorflow as tf

from rbm.learning.constant_learning_rate import ConstantLearningRate
from rbm.train.task.persistent_task import PersistentTask
from rbm.train.task.rbm_mensurate_task import RBMMeasureTask
from rbm.train.task.summary_task import SummaryTask
from rbm.train.task.task import Task
from rbm.train.trainer import Trainer
from rbm.util.util import square, mean, Σ


class MeasureQuantitativeTask(Task):

    def init(self, trainer: Trainer, session: tf.Session):
        dataset = tf.constant(trainer.dataset.T.values, dtype=tf.float32)
        reconstructed = trainer.model.gibbs_step(dataset)

        with tf.name_scope('reconstruction'):
            tf.summary.scalar('error', square(mean(tf.abs(dataset - reconstructed))))
            tf.summary.scalar('hamming', self.hamming_distance(dataset, reconstructed))

    def hamming_distance(self, a, b):
        return Σ(tf.abs(a - b))


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


def train(dataset, batch_size=10, epochs=100, hidden_size=100, learning_rate=None, regularization=None, sampling_method=None, persist=False):
    """
    # Batch_size = 10 or 100
    # https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf
    """
    from rbm.rbm import RBM

    tf.set_random_seed(42)

    total_elements, size_element = dataset.shape

    # movies_size=1,
    # ratings_size=size_element,

    rbm = RBM(
        visible_size=size_element,
        hidden_size=hidden_size,
        regularization=regularization,
        learning_rate=ConstantLearningRate(10**-2),
        sampling_method=sampling_method,
    )

    trainer = Trainer(rbm, dataset, batch_size=batch_size)
    trainer.stopping_criteria.append(lambda current_epoch: current_epoch > epochs)

    log = f"results/logs/{rbm}/{time.time()}"

    trainer.tasks.append(MeasureQuantitativeTask())
    trainer.tasks.append(RBMMeasureTask())
    trainer.tasks.append(SummaryTask(log=log))
    #trainer.tasks.append(BeholderTask(log='results/logs'))

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


train(
    bag_of_plugins,
    batch_size=10,
    epochs=200,
    learning_rate=ConstantLearningRate(10**-2),
)
