import time
from collections import OrderedDict
from itertools import product
from typing import Dict, Iterable

import pandas as pd
import tensorflow as tf

from rbm.rbm import RBM
from rbm.rbmcf import RBMCF
from rbm.train.task.beholder_task import BeholderTask
from rbm.train.task.persistent_task import PersistentTask
from rbm.train.task.rbm_inspect_histograms_task import RBMInspectHistogramsTask
from rbm.train.task.rbm_inspect_scalars_task import RBMInspectScalarsTask
from rbm.train.task.rbm_measure_task import RBMMeasureTask
from rbm.train.task.rbmcf_measure_task import RBMCFMeasureTask
from rbm.train.task.summary_task import SummaryTask
from rbm.train.trainer import Trainer
from rbm.util.embedding import reasonable_visible_bias


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


def train(kfold: str,
          data_train: pd.DataFrame, data_validation: pd.DataFrame,
          batch_size=10, epochs=100, hidden_size=100,
          model_class=None,
          learning_rate=None, momentum=0,
          regularization=None, sampling_method=None,
          persist=False, log_epoch_step=10, log_every_epoch=None,
          save_path='./results/model'):
    """
    # Batch_size = 10 or 100
    # https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf
    """

    tf.set_random_seed(42)

    total_elements, size_element = data_train.shape
    b_v = reasonable_visible_bias(data_train)

    if model_class == RBM:
        rbm = RBM(
            visible_size=size_element,
            hidden_size=hidden_size,
            regularization=regularization,
            learning_rate=learning_rate,
            sampling_method=sampling_method,
            momentum=momentum,
            b_v=b_v,
        )
    elif model_class == RBMCF:
        rbm = RBMCF(
            movies_size=6,
            ratings_size=int(size_element / 6),
            hidden_size=hidden_size,
            regularization=regularization,
            learning_rate=learning_rate,
            sampling_method=sampling_method,
            momentum=momentum,
            b_v=b_v,
        )
    else:
        raise Exception('Invalid RBM')

    trainer = Trainer(rbm, data_train, batch_size=batch_size)

    trainer.stopping_criteria.append(lambda current_epoch: current_epoch > epochs)

    log = f"results/logs/kfold={kfold}/batch_size={batch_size}/{rbm}/{time.time()}"

    #trainer.tasks.append(RBMInspectScalarsTask())
    #trainer.tasks.append(RBMInspectHistogramsTask())

    if model_class == RBM:
        task = RBMMeasureTask(
            movies_size=6,
            ratings_size=int(size_element / 6),
            data_train=data_train,
            data_validation=data_validation
        )
        trainer.tasks.append(task)

    if model_class == RBMCF:
        task = RBMCFMeasureTask(
            data_train=data_train,
            data_validation=data_validation,
        )
        trainer.tasks.append(task)

    trainer.tasks.append(SummaryTask(log=log, epoch_step=log_epoch_step, every_epoch=log_every_epoch))
    #trainer.tasks.append(BeholderTask(log='results/logs'))
    path = None

    if persist:
        path = f"{save_path}/kfold={kfold.replace('/', '+')}+batch_size={batch_size}+{rbm.__str__().replace('/', '+')}/rbm.ckpt"
        trainer.tasks.append(PersistentTask(path=path, save_after_every=int(1e100)))

    print('Training', log)
    trainer.train()

    return rbm, path
