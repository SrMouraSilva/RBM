import math
from typing import Iterable, Iterator, Sized, Union

import numpy as np


class Batch(Iterable):

    def __init__(self, data_x: Union[Iterable, Sized], data_y: Union[Iterable, Sized], start: int, size: int):
        self.data_x = data_x
        self.data_y = data_y if data_y is not None else np.array([])

        self.start = start
        self.current = start
        self.size = size
        self.total = math.ceil(len(data_x)/size)

    def __iter__(self) -> Iterator:
        self.current = self.start
        return self

    def __next__(self):
        """
        Implements as iterate
        :return:
        """
        i = self.current*self.size
        j = (self.current+1)*self.size

        data_x = self.data_x[i:j]
        data_y = self.data_y[i:j]

        if len(data_x) == 0:
            raise StopIteration()

        self.current += 1
        return data_x.T, data_y.T
