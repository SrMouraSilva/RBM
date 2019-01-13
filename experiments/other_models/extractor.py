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
        z = x.copy()
        z['y'] = y
        z += 1
        z.columns = list(range(1, 6)) + ['y']
        #z = z.reset_index('name')
        #del z['name']
        z.to_csv(f'data/train-{self.counter//6}_{self.counter%6}.csv', index=False)
        self.counter += 1

    def predict(self, x):
        z = x.copy()
        z.columns = list(range(1, 6))
        z.to_csv(f'data/test-{self.counter // 6}_{self.counter % 6}.csv', index=False)

        return np.zeros(x.shape[0]) - 1
