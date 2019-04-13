import numpy as np
import pandas as pd
from sklearn.utils import shuffle

from rbm.train.kfold_cross_validation import KFoldCrossValidation


def load_data(path='data'):
    return pd.read_csv(f'{path}/patches-filtered.csv', sep=",", index_col=['id', 'name']).astype(np.int32)


def load_data_one_hot_encoding(path='data'):
    return pd.read_csv(f'{path}/patches-one-hot-encoding.csv', sep=",", index_col=['index', 'id'], dtype=np.float32)


def load_data_categories(path='data'):
    return pd.read_csv(f'{path}/plugins_categories_simplified.csv', sep=",", index_col=['id'])


def get_specific_fold(fold, data, n_splits=5, random_state=42):
    data_shuffled = shuffle(data, random_state=random_state)
    cross_validation = KFoldCrossValidation(data=data_shuffled, n_splits=n_splits, random_state=42, shuffle=False)

    for i, train, test in cross_validation.split():
        if i == fold:
            return train, test
