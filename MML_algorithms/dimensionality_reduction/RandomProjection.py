import numpy as np
from numpy.random import RandomState
from ..utils import check_data

class RandomProjection:
    def __init__(self, num_components=None, random_state=None):
        '''

        :param num_components:
        :param random_state:
        '''
        if num_components is None:
            self.num_components = 2
        else:
            self.num_components = num_components
        self.num_samples = None
        self.num_features = None

        self.random_state = random_state
        self.components = None

    def fit(self, X):
        '''

        :param X:
        :return:
        '''
        check_data(X)
        self.num_samples, self.num_features = X.shape
        self.components = self._gaussian_matrix()
        return self

    def transform(self, X):
        '''

        :param X:
        :return:
        '''
        check_data(X)
        X_projected = X @ self.components.T
        return X_projected

    def fit_transform(self, X):
        '''

        :param X:
        :return:
        '''
        self.fit(X)
        return self.transform(X)

    def _gaussian_matrix(self):
        '''
        Creates

        :return: a gaussian matrix of dimension num_components-by-num_features
        '''
        random_generator = RandomState(self.random_state)
        return random_generator.normal(loc=0.0,
                                       scale=1/np.sqrt(self.num_components),
                                       size=(self.num_components, self.num_features))

