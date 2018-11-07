import os

import numpy as np
import pandas as pd
from tensorflow.keras.datasets import mnist

from experiments.experiment import Experiment
from rbm.drbm import DRBM
from rbm.learning.constant_learning_rate import ConstantLearningRate
from rbm.sampling.contrastive_divergence import ContrastiveDivergence


def one_hot(a, num_classes):
    return np.squeeze(np.eye(num_classes)[a.reshape(-1)])


(x_train, y_train), (x_test, y_test) = mnist.load_data(f'{os.getcwd()}/data/tmp/mnist.npz')

# thresholding at 1/2
# http://www.cs.utoronto.ca/~tijmen/pcd/pcd.pdf
x_train, x_test = x_train > 127, x_test > 127

x_train = x_train.reshape([x_train.shape[0], x_train.shape[1]*x_train.shape[2]])
x_train = pd.DataFrame.from_records(x_train)

y_train = one_hot(y_train, 10)
y_train = pd.DataFrame.from_records(y_train)

cross_validation = {
    'data_x': [x_train],
    'data_y': [y_train],
    'batch_size': [10],
    'hidden_size': [2],
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
        DRBM
    ]
}

experiment = Experiment()
experiment.train(cross_validation)
