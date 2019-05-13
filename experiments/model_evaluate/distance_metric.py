import numpy as np

from sklearn.metrics.pairwise import cosine_similarity


def cosine_distance(X, Y):
    """
    http://www.facom.ufu.br/~elaine/disc/MFCD/Aula7-MedidasDistancia.pdf
    """
    if len(np.asarray(X).shape) < 2:
        X = [X]
    if len(np.asarray(Y).shape) < 2:
        Y = [Y]

    return 1 - cosine_similarity(X, Y)
