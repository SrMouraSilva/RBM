import numpy as np
from experiments.other_models.other_model import OtherModel


class ExtractorModel(OtherModel):
    """
    This model only extract the data and save in files
    """

    def __init__(self):
        super().__init__()
        self.counter = 0

    def fit(self, x, y):
        z = self._prepare_data(x, y)
        z.to_csv(f'other_models/data/train-{self.counter//6}_{self.counter%6}.csv', index=False, header=False)
        self.counter += 1

    def predict(self, x, y=None, comment=''):
        z = self._prepare_data(x, y)
        counter = self.counter - 1
        z.to_csv(f'other_models/data/test-{comment}-{counter // 6}_{counter % 6}.csv', index=False, header=False)

        return np.zeros(x.shape[0]) - 1

    def _prepare_data(self, x, y):
        z = x.copy()
        z['y'] = y
        z += 1
        z.columns = list(range(1, 6)) + ['y']

        return z
