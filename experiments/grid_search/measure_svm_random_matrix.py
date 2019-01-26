import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.utils import shuffle

from rbm.train.kfold_elements import KFoldElements

RANDOM_STATE = 42
np.random.seed(seed=RANDOM_STATE)
COLUMNS = 6
RANDOM_MATRIX = np.random.rand(COLUMNS-1, COLUMNS-1)
METHOD_NAME = 'SVC-random-matrix'

data = pd.read_csv('../data/pedalboard-plugin.csv', sep=",", index_col=['id', 'name'])

data_shuffled = shuffle(data, random_state=RANDOM_STATE)
kfolds_training_test = KFoldElements(data=data_shuffled, n_splits=5, random_state=RANDOM_STATE, shuffle=False)


param_grid = {
    'C': [1e-5, 10e-3, 10e-1, 10e1, 10e3, 10e5],
    'gamma': [1e-5, 10e-3, 10e-1, 10e1, 10e3, 10e5, 'scale'],
    'kernel': ['rbf']
}


def split_x_y(data, test_column_index):
    columns = [f'plugin{i}' for i in range(1, COLUMNS + 1)]
    train_columns = columns[0:test_column_index] + columns[test_column_index + 1:COLUMNS + 1]
    test_column = f'plugin{test_column_index + 1}'

    return data[train_columns], data[test_column]


dataframes = []
for i, original_training, test in kfolds_training_test.split():
    #kfolds_training_validation = KFoldElements(data=original_training, n_splits=2, random_state=RANDOM_STATE, shuffle=False)
    #for j, training, validation in kfolds_training_validation.split():
    for column in range(COLUMNS):
        dataframe = {
            'kfold-test': i,
            #'kfold-validation': j,
            'column': column,
            'is_test': False,
            'evaluation': 'train',
            'evaluation_method': METHOD_NAME
        }

        X, y = split_x_y(original_training, column)
        X = X @ RANDOM_MATRIX

        clf = GridSearchCV(svm.SVC(), param_grid, cv=2, n_jobs=-1, scoring=f'accuracy')
        clf.fit(X, y)

        for c in ['split0_train_score', 'split1_train_score', 'mean_train_score', 'std_train_score', 'mean_test_score', 'std_test_score', 'params']:
            dataframe[c] = clf.cv_results_[c]

        dataframes.append(pd.DataFrame(dataframe))


pd.concat(dataframes).to_csv(f'../other_models/data_search_parameter/{METHOD_NAME}.csv')
