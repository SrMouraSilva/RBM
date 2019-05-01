class TestDefinition:
    """
    A test is defined by a model, the parameters that will be applied and the split method

    :param y_column: None for check all columns
                     Int for index of the specific column
    """
    def __init__(self, model, params, split_method, refit=None, y_column=None):
        self.model = model
        self.params = params
        self.split_method = split_method
        self.refit = refit
        self.y_column = y_column

    def test_all_columns(self):
        return self.y_column is None

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
