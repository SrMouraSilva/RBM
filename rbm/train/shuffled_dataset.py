import math
from typing import Iterable, Iterator, Sized, Union

import tensorflow as tf


class ShuffledDataset(Iterable):

    def __init__(self, data: Union[Iterable, Sized], batch_size: int):
        self.batch_size = batch_size

        self.current_batch = 0
        self.total_batches = math.ceil(len(data)/batch_size)

        total_elements = data.shape[0]

        self.dataset = tf.data.Dataset.from_tensor_slices(data) \
            .shuffle(buffer_size=total_elements) \
            .batch(batch_size) \
            .repeat(None)

        self._iterator = self.dataset.make_one_shot_iterator()

    def get_next(self):
        return self._iterator.get_next().T

    def __iter__(self) -> Iterator:
        self.current = 0
        return self

    def __next__(self):
        if self.current >= self.total_batches:
            raise StopIteration()

        self.current += 1

        return self.current
