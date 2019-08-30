import numpy as np
from numpy.random.mtrand import RandomState

from rbm.util.embedding import one_hot_encoding


class NeuralAutoRegressiveNetwork:

    def __init__(self, visible_size, visible_values_size, hidden_size, seed=None):
        seed = seed if seed is not None else 42
        self.random: RandomState = RandomState(seed)

        self.visible_size = visible_size
        self.visible_values_size = visible_values_size
        self.hidden_size = hidden_size

        hidden_layers = self.visible_size
        V_dimension = (hidden_layers, self.visible_size, self.hidden_size, self.visible_values_size)
        self.V = self.random.rand(*V_dimension)
        W_dimension = (self.hidden_size, self.visible_values_size, self.visible_size)
        self.W = self.random.rand(*W_dimension)

        self.C = np.zeros((self.hidden_size, hidden_layers))
        self.B = np.zeros((self.visible_values_size, self.visible_size))

    def Vlej(self, j):
        """
        Weight associated with the j-th hidden layer and all k<j visible units
        :param j:
        :return:
        """
        return self.V[j, :j, :, :]

    def Cj(self, j: int):
        """
        :return: j-th vector column of C that corresponds the bias of the j-th hidden column
        """
        return self.C[:, [j]]

    def Bi(self, i: int):
        """
        :return: i-th vector column of C that corresponds the bias of the i-th output column
        """
        return self.B[:, [i]]

    def first_columns(self, matrix, j: int):
        """
        :return: All the (j-1)-th vector columns of the matrix
        """
        return matrix[:, :j]

    def first_tables(self, matrix, j: int):
        """
        :return: All the (j-1)-th first 2d matrix of the 3d matrix
        """
        return matrix[:, :, :j]

    def calculate(self, data):
        H = np.zeros((self.hidden_size, self.visible_size))
        for j in range(1, self.visible_size):
            H[:, j] = self.Hj(j, data).T

        G = np.zeros((self.hidden_size, self.visible_size))
        for i in range(self.visible_size):
            G[:, i] = self.Gi(i, H)

    def Hj(self, j, data):
        Vlej = self.Vlej(j)
        Zlej = self.first_columns(data, j)

        print('V', Vlej.shape)
        print('Z', Zlej.shape)
        print('C', self.Cj(j).shape)
        #print(Vlej.T @ Zlej)

        '''
        print('Produtos')
        try:
            print(1, (Vlej @ Zlej).shape)
        except:
            pass
        try:
            print(2, (Vlej @ Zlej.T).shape)
        except:
            pass
        try:
            print(3, (Vlej.T @ Zlej).shape)
        except:
            pass
        try:
            print(4, (Vlej.T @ Zlej.T).shape)
        except:
            pass
        '''

        print('Product shape', (Vlej @ Zlej).shape)
        print('Product shape', np.sum(Vlej @ Zlej, axis=0).shape)
        print('Product shape', np.sum(Vlej @ Zlej, axis=1).shape)
        print('Product shape', np.sum(Vlej @ Zlej, axis=2).shape)
        print('Product shape', np.sum(Vlej @ Zlej, axis=-1).shape)
        return np.tanh(self.Cj(j) + Vlej @ Zlej).reshape(self.hidden_size, 1)

    def Gi(self, i, H):
        Wleqj = self.first_tables(self.W, i+1)
        Hleqj = self.first_columns(H, i+1)

        return np.tanh(self.Bi(i) + Wleqj @ Hleqj)


data_samples = 5
visible_size, visible_values_size, hidden_size = 11, 3, 7
NARNia = NeuralAutoRegressiveNetwork(visible_size, visible_values_size, hidden_size)

NARNia.random.randint(0, visible_values_size, size=visible_size)
data = one_hot_encoding(NARNia.random.randint(0, visible_values_size, size=visible_size), depth=visible_values_size).T

print(data)
NARNia.calculate(data)
