import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(title: str, y_originals: list, y_predicts: list, columns_order, columns_names):
    matrix = np.zeros(shape=[117, 117])

    for y, y_predict in zip(y_originals, y_predicts):
        matrix += confusion_matrix(y, y_predict, labels=range(117))

    matrix = pd.DataFrame(matrix)
    matrix = matrix[columns_order]
    matrix = matrix.reindex(columns_order)
    matrix.columns = columns_names
    matrix.index = columns_names

    # Remove y_originals not used
    #matrix = matrix[matrix.sum(axis=1) != 0]
    # Remove y_predicts not used
    #matrix = matrix.T[matrix.T.sum(axis=1) != 0].T

    # Blank not recommended values
    mask = matrix == 0

    ax = sns.heatmap(data=matrix, cmap="YlGnBu", center=1, mask=mask, square=True)
    ax.set_xlabel('predict')
    ax.set_ylabel('test')
    ax.set_title(title)

    return ax
