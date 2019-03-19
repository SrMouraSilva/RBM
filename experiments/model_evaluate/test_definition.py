class TestDefinition:
    """
    A test is defined by a model, the parameters that will be applied and the split method
    """
    def __init__(self, model, params, split_method):
        self.model = model
        self.params = params
        self.split_method = split_method

    def __str__(self):
        return f'{self.model.__name__}-{self.split_method.__name__}-{self.params}'
