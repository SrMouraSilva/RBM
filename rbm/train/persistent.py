from abc import abstractmethod, ABCMeta


class Persistent(metaclass=ABCMeta):

    @abstractmethod
    def save(self, path, session):
        pass

    @abstractmethod
    def load(self, path, session):
        pass
