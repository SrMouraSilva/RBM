import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(title: str, y_originals: list, y_predicts: list, columns_order, columns_names, absolute=False, group_by_category=False):
    matrix = np.zeros(shape=[117, 117])

    for y, y_predict in zip(y_originals, y_predicts):
        matrix += confusion_matrix(y, y_predict, labels=range(117))

    matrix = pd.DataFrame(matrix)
    matrix = matrix[columns_order]
    matrix = matrix.reindex(columns_order)
    matrix.columns = columns_names
    matrix.index = columns_names

    if not absolute:
        #matrix = (matrix.T / matrix.sum(axis=1).T).T
        matrix = matrix / matrix.sum(axis=1).values.reshape((-1, 1))
        #matrix = matrix / matrix.sum(axis=1)

    # Remove y_originals not used
    #matrix = matrix[matrix.sum(axis=1) != 0]
    # Remove y_predicts not used
    #matrix = matrix.T[matrix.T.sum(axis=1) != 0].T

    if group_by_category:
        matrix = matrix.reset_index().groupby('category', sort=False).sum() \
                     .T.reset_index().groupby('category', sort=False).sum().T

    # Blank not recommended values
    mask = matrix == 0

    ax = sns.heatmap(data=matrix, cmap="YlGnBu", center=.0001, mask=mask, square=True)
    ax.set_xlabel('predict')
    ax.set_ylabel('true')
    ax.set_title(title)

    return ax
