import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.random_projection import GaussianRandomProjection
from sklearn.utils import shuffle

from rbm.train.kfold_elements import KFoldElements


RANDOM_STATE = 42
np.random.seed(seed=RANDOM_STATE)
COLUMNS = 6
CATEGORIES = 117
METHOD_NAME = 'SVC-bag_of_words_gaussian_random_projection'
transformer = GaussianRandomProjection(n_components=50)#eps=.5)

data = pd.read_csv('../data/pedalboard-plugin-full-bag-of-words.csv', sep=",", index_col=['id', 'index'])

data_shuffled = shuffle(data, random_state=RANDOM_STATE)
kfolds_training_test = KFoldElements(data=data_shuffled, n_splits=5, random_state=RANDOM_STATE, shuffle=False)

metrics = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']


param_grid = {
    'C': [1e-5, 10e-3, 10e-1, 10e1, 10e3, 10e5],
    'gamma': [1e-5, 10e-3, 10e-1, 10e1, 10e3, 10e5, 'scale'],
    'kernel': ['rbf']
}


def split_x_y(data, test_column_index):
    train_columns = list(data.columns[0:test_column_index*CATEGORIES]) \
                  + list(data.columns[(test_column_index+1)*CATEGORIES: COLUMNS*CATEGORIES + 1])
    test_column = data.columns[test_column_index*CATEGORIES:(test_column_index+1)*CATEGORIES]

    return data[train_columns], np.argmax(data[test_column].values, axis=1)


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
        X = transformer.fit_transform(X)

        clf = GridSearchCV(svm.SVC(), param_grid, cv=2, n_jobs=-1, scoring=metrics, refit=metrics[0])
        clf.fit(X, y)

        dataframe.update(clf.cv_results_)

        dataframes.append(pd.DataFrame(dataframe))


pd.concat(dataframes).to_csv(f'../evaluate_results/_old/data_search_parameter/{METHOD_NAME}.csv')
