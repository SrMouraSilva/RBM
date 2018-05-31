from typing import Iterable, Iterator


class Batch(Iterable):

    def __init__(self, data: Iterable, start: int, size: int):
        self.data = data

        self.start = start
        self.current = start
        self.size = size

    def __iter__(self) -> Iterator:
        #self.current = self.start
        return self

    def __next__(self):
        """
        Implements as iterate
        :return:
        """
        i = self.current*self.size
        j = (self.current+1)*self.size

        data = self.data[i:j]

        if len(data) == 0:
            raise StopIteration()

        return data.T
