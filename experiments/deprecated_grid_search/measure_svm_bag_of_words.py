import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.utils import shuffle

from experiments.model_evaluate.evaluate_method import mrr_score_function
from experiments.model_evaluate.split_method import split_with_bag_of_words_function
from rbm.train.kfold_elements import KFoldElements

RANDOM_STATE = 42
np.random.seed(seed=RANDOM_STATE)
COLUMNS = 6
CATEGORIES = 117
METHOD_NAME = 'SVC-bag_of_words'

data = pd.read_csv('../data/pedalboard-plugin.csv', sep=",", index_col=['id', 'name'])

data_shuffled = shuffle(data, random_state=RANDOM_STATE)
kfolds_training_test = KFoldElements(data=data_shuffled, n_splits=5, random_state=RANDOM_STATE, shuffle=False)

#metrics = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
metrics = {
    'accuracy': 'accuracy',
    'precision_weighted': 'precision_weighted',
    'recall_weighted': 'recall_weighted',
    'f1_weighted': 'f1_weighted',
    'mrr': mrr_score_function(CATEGORIES)
}


param_grid = {
    'C': [1e-5, 10e-3, 10e-1, 10e1, 10e3, 10e5],
    'kernel': ['linear']
}


def split_x_y(data, test_column_index):
    train_columns = list(data.columns[0:test_column_index*CATEGORIES]) \
                  + list(data.columns[(test_column_index+1)*CATEGORIES: COLUMNS*CATEGORIES + 1])
    test_column = data.columns[test_column_index*CATEGORIES:(test_column_index+1)*CATEGORIES]

    return data[train_columns], np.argmax(data[test_column].values, axis=1)


for i, original_training, test in kfolds_training_test.split():
    for column in range(COLUMNS):
        dataframe = {
            'kfold-test': i,
            'column': column,
            'evaluation_method': METHOD_NAME
        }

        X, y = split_with_bag_of_words_function(CATEGORIES)(original_training, column)

        clf = GridSearchCV(svm.SVC(), param_grid, cv=2, n_jobs=-1, scoring=metrics, refit=False, return_train_score=True)
        clf.fit(X, y)

        dataframe.update(clf.cv_results_)

        pd.DataFrame(dataframe).to_csv(f'../evaluate_results/{METHOD_NAME}+{i}-{column}.csv')
