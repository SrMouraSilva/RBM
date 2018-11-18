from abc import ABCMeta, abstractmethod

from rbm.rbm import RBM


class Predictor(metaclass=ABCMeta):

    def __init__(self, model: RBM, movie_size: int, rating_size: int):
        self.model = model

        self.movie_size = movie_size
        self.rating_size = rating_size

        self.shape_softmax = [-1, movie_size, rating_size]
        self.shape_visibleT = [-1, model.visible_size]

    @abstractmethod
    def predict(self, v):
        pass
