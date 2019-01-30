import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.random_projection import GaussianRandomProjection
from sklearn.utils import shuffle

from rbm.train.kfold_elements import KFoldElements

print('Estava testando calcular usando GridSearchCV, no lugar do calculador que tinha feito manualmente')
exit()

RANDOM_STATE = 42
np.random.seed(seed=RANDOM_STATE)
COLUMNS = 6
CATEGORIES = 117
METHOD_NAME = 'SVC-bag_of_words_gaussian_random_projection 2'
transformer = GaussianRandomProjection(n_components=50)#eps=.5)

data = pd.read_csv('../data/pedalboard-plugin-full-bag-of-words.csv', sep=",", index_col=['id', 'index'])

data_shuffled = shuffle(data, random_state=RANDOM_STATE)


param_grid = {
    'C': [10000],
    'gamma': [1e-5],
    'kernel': ['rbf']
}


def split_x_y(data, test_column_index):
    train_columns = list(data.columns[0:test_column_index*CATEGORIES]) \
                  + list(data.columns[(test_column_index+1)*CATEGORIES: COLUMNS*CATEGORIES + 1])
    test_column = data.columns[test_column_index*CATEGORIES:(test_column_index+1)*CATEGORIES]

    return data[train_columns], np.argmax(data[test_column].values, axis=1)

dataframes=[]

for column in range(COLUMNS):
    dataframe = {
        'kfold-test': 0,
        #'kfold-validation': j,
        'column': column,
        'is_test': False,
        'evaluation': 'train',
        'evaluation_method': METHOD_NAME
    }

    X, y = split_x_y(data_shuffled, column)
    X = transformer.fit_transform(X)

    clf = GridSearchCV(svm.SVC(), param_grid, cv=5, n_jobs=-1, scoring=f'accuracy')
    clf.fit(X, y)

    for c in ['mean_train_score', 'mean_test_score', 'params']:
        dataframe[c] = clf.cv_results_[c]

    dataframes.append(pd.DataFrame(dataframe))


pd.concat(dataframes).to_csv(f'../evaluate_results/_old/data_search_parameter/{METHOD_NAME}.csv')
