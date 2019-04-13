import time

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils import shuffle
from tqdm import tqdm

from rbm.rbmcf import RBMCF
from rbm.sampling.contrastive_divergence import ContrastiveDivergence
from rbm.train.kfold_cross_validation import KFoldCrossValidation
from rbm.train.task.persistent_task import PersistentTask
from rbm.train.task.rbm_inspect_scalars_task import RBMInspectScalarsTask
from rbm.train.task.rbmcf_measure_task import RBMCFMeasureTask
from rbm.train.task.summary_task import SummaryTask
from rbm.train.trainer import Trainer


def read_data(path, index_col=None):
    if index_col is None:
        index_col = ['index', 'id']

    return pd.read_csv(path, sep=",", index_col=index_col, dtype=np.float32)


def model(size_element=702):
    return RBMCF(
        movies_size=6,
        ratings_size=int(size_element / 6),
        hidden_size=1000,
        regularization=None,
        learning_rate=0.2,
        sampling_method=ContrastiveDivergence(1),
        momentum=1
    )


def train_model(rbm, data_train, data_test, log_path, model_path):
    trainer = Trainer(rbm, data_train, batch_size=BATCH_SIZE)

    trainer.stopping_criteria.append(lambda current_epoch: current_epoch > 1000)

    trainer.tasks.append(RBMInspectScalarsTask())
    trainer.tasks.append(RBMCFMeasureTask(
        data_train=data_train,
        data_validation=data_test,
    ))

    trainer.tasks.append(SummaryTask(log=log_path, epoch_step=10))
    trainer.tasks.append(PersistentTask(path=model_path))

    print('Training', log_path)
    trainer.train()


train = False

rbm = model()

BATCH_SIZE = 10
log_path = f"results/logs/kfold={0}/batch_size={BATCH_SIZE}/{rbm}/{time.time()}"
model_path = f"./results/model/batch_size={BATCH_SIZE}+{rbm.__str__().replace('/', '+')}/rbm.ckpt"

original_bag_of_plugins = read_data('data/pedalboard-plugin-full-bag-of-words.csv')

if train:
    bag_of_plugins = shuffle(original_bag_of_plugins, random_state=42)
    kfolds_training_test = KFoldCrossValidation(data=bag_of_plugins, n_splits=5, random_state=42, shuffle=False)

    for i, original_training, test in kfolds_training_test.split():
        kfolds_training_validation = KFoldCrossValidation(data=original_training, n_splits=2, random_state=42, shuffle=False)

        train_model(rbm, original_training, test, log_path, model_path)
        break

else:
    with tf.Session() as session:
        # To remove index
        database = original_bag_of_plugins.reset_index('index')
        del database['index']

        # To load the selected presets
        presets = {
            'Pink Floyd': [6287, 6288, 6492, 7254, 9376],
            'Metallica': [9504, 8801, 8415, 8339, 8139],
            'Jazz': [8791, 7673, 7986],
            'Beatles': [7973, 6581, 6578, 5358, 6584],
            'Dire Straits': [7559, 7005, 8145, 8660],
            'Red Hot': [9568, 9375, 9570, 9569, 9575],
        }
        base_colors = {
            'Pink Floyd': 'maroon',
            'Metallica': 'y',
            'Jazz': 'cyan',
            'Beatles': 'g',
            'Dire Straits': 'b',
            'Red Hot': 'r',
        }

        # Load model
        rbm.load(session, model_path)

        # Calc X and y
        X = []
        y = []
        colors = []

        for k, v in presets.items():
            rows = database.loc[v].dropna()

            print(k)
            for i in tqdm(range(200)):
                X.append(rbm.sample_h_given_v(rows.T.values).eval())
                y += [k] * len(v)
                colors += [base_colors[k]] * len(v)

        X = np.concatenate([x.T for x in X])
        y = np.array(y)
        colors = np.array(colors)
        #colors = np.array(['r' if yi == 'Pink Floyd' else 'b' for yi in y])

        from yellowbrick.features.pca import PCADecomposition

        visualizer = PCADecomposition(scale=True, color=colors)
        visualizer.fit_transform(X, y)
        visualizer.poof()

        from yellowbrick.features.manifold import Manifold

        visualizer = Manifold(manifold='tsne', target='discrete')
        visualizer.fit_transform(X, y)
        visualizer.poof()
