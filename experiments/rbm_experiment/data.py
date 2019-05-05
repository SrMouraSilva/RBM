from typing import Tuple

import numpy as np


class Data:

    def __init__(self, data: np.array, total_movies: int):
        """
        :param data: Expected a Matrix of line vectors
        """
        batch_size, total_columns = data.shape

        self.data = data
        self.total_movies = total_movies
        self.rating_size = total_columns // total_movies

    def to_missing_movies(self, movies: [int]) -> Tuple[np.array, np.array]:
        X = self.hide_movies(movies)
        y = [self.extract_movie(movie) for movie in movies]

        return X, y

    def hide_movies(self, movies: [int]) -> np.ndarray:
        """
        Transform column to missing data
        """
        X = self.data.copy()

        for movie in movies:
            i, j = self._axis(movie)
            X[:, i:j] = 0

        return X

    def extract_movie(self, movie: int) -> np.ndarray:
        i, j = self._axis(movie)

        return self.data[:, i:j]

    def _axis(self, column: int) -> Tuple[int, int]:
        i = column * self.rating_size
        j = (column + 1) * self.rating_size

        return i, j
