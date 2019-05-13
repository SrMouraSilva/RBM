from typing import Type

from sklearn.base import TransformerMixin
from sklearn.pipeline import make_pipeline

from experiments.model_evaluate.evaluate_method.evaluate_method import ScikitLearnClassifierModel
from experiments.model_evaluate.split_method import SplitFunction
from experiments.transform.no_transform import NoTransform


class TestDefinition:
    """
    A test is defined by a model, the parameters that will be applied and the split method

    :param y_column: None for check all columns
                     Int for index of the specific column
    """
    def __init__(self, model: Type[ScikitLearnClassifierModel], params: dict, split_method: SplitFunction, refit: int = None, y_column: int=None, transform: TransformerMixin=None):
        self.model = model
        self.params = params
        self.split_method = split_method
        self.refit = refit
        self.y_column = y_column

        if transform is None:
            self.transform = NoTransform()
        else:
            self.transform = transform

    def test_all_columns(self):
        return self.y_column is None

    @property
    def pipeline_model(self):
        return make_pipeline(self.transform, self.model())

    @property
    def pipeline_params(self):
        model_name = self.model.__name__.lower()
        data = {}
        for k, v in self.params.items():
            data[f'{model_name}__{k}'] = v

        return data

    def __dict__(self):
        return {
            'model': self.model.__name__,
            'params': self.params.__str__(),
            'split_method': self.split_method.__name__,
            'refit': self.refit
        }

    def __str__(self):
        string = f'{self.model.__name__}-{self.split_method.__name__}-{self.params}-{self.refit}'

        if not self.test_all_columns():
            string += f'-{self.y_column}'

        return string
