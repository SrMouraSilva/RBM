import pandas as pd
from sklearn.utils import shuffle
from tqdm import tqdm

from experiments.other_models.other_model import OtherModel
from rbm.train.kfold_elements import KFoldElements


class ModelEvaluate:

    def __init__(self, random_state=42, columns=6):
        self.random_state = random_state
        self.columns = columns

    def _read_data(self, path, index_col=None):
        if index_col is None:
            index_col = ['id', 'name']

        return pd.read_csv(path, sep=",", index_col=index_col)

    def evaluate(self, model):
        data = self._read_data('../data/pedalboard-plugin.csv')

        data_shuffled = shuffle(data, random_state=self.random_state)

        kfolds_training_test = KFoldElements(data=data_shuffled, n_splits=5, random_state=self.random_state, shuffle=False)

        data = pd.DataFrame(columns=['kfold-test', 'kfold-validation', 'model', 0, 1, 2, 3, 4, 5, 'is_test', 'evaluation'])

        for i, original_training, test in tqdm(kfolds_training_test.split(), desc='5-fold', total=5):
            kfolds_training_validation = KFoldElements(data=original_training, n_splits=2, random_state=self.random_state, shuffle=False)

            #for j, training, validation in kfolds_training_validation.split():
            #    evaluates = self.evaluate_by_column(model, training, validation)
            #    for evaluate in evaluates:
            #        evaluate['kfold-test'] = i
            #        evaluate['kfold-validation'] = j
            #        evaluate['is_test'] = False
            #
            #    data = data.append(evaluates, ignore_index=True)

            evaluates = self.evaluate_by_column(model, original_training, test)
            for evaluate in evaluates:
                evaluate['kfold-test'] = i
                evaluate['kfold-validation'] = None
                evaluate['is_test'] = True

            data = data.append(evaluates, ignore_index=True)

        return data

    def evaluate_by_column(self, model: OtherModel, training, test):
        values_train = {
            'model': model.__class__.__name__,
            'evaluation': 'train'
        }
        values_test = {
            'model': model.__class__.__name__,
            'evaluation': 'test'
        }

        for column in tqdm(range(self.columns), desc='Column evaluate'):
            x_train, y_train = self._split_x_y(training, column)
            x_test, y_expected = self._split_x_y(test, column)

            model.initialize()
            model.fit(x_train, y_train)

            y_train_generated = model.predict(x_train)
            y_test_generated = model.predict(x_test)

            total_equals_train = sum(y_train.values == y_train_generated)
            total_equals_test = sum(y_expected.values == y_test_generated)

            values_train[column] = total_equals_train/len(y_train_generated)
            values_test[column] = total_equals_test/len(y_test_generated)

        return [values_train, values_test]

    def _split_x_y(self, data, test_column_index):
        columns = [f'plugin{i}' for i in range(1, self.columns+1)]
        train_columns = columns[0:test_column_index] + columns[test_column_index+1:self.columns+1]
        test_column = f'plugin{test_column_index+1}'

        return data[train_columns], data[test_column]
