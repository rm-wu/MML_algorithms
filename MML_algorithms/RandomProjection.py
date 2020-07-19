import numpy as np
from numpy.random import RandomState
from ..utils import check_data


class GaussianRandomProjection:
    def __init__(self, num_components=None, random_state=None):
        """
        Random Projections implementation
        :param num_components:
            number of components of the projection
        :param random_state:
            random state to fix the initial state of the random
        """
        if num_components is None:
            self.num_components = 2
        else:
            self.num_components = num_components
        self.num_samples = None
        self.num_features = None

        self.random_state = random_state
        self.components = None

    def fit(self, X):
        """
        Takes as input a data matrix X and fits the model

        :param X: np.ndarray
            Data to project
        :return:
            RandomProjection object fitted
        """

        check_data(X)
        self.num_samples, self.num_features = X.shape
        self.components = self._gaussian_matrix()
        return self

    def transform(self, X):
        """
        Takes as input a data matrix X and returns its projection into the random matrix

        :param X: np.ndarray
            Data to project
        :return:
            Projected data
        """
        check_data(X)
        X_projected = X @ self.components.T
        return X_projected

    def fit_transform(self, X):
        ''' Takes as input a data matrix X and returns its projection into the random matrix

        :param X: np.ndarray
            Data to project
        :return:
            Projected data
        '''
        self.fit(X)
        return self.transform(X)

    def _gaussian_matrix(self):
        '''
        Creates a Gaussian matrix with mean 0.0 and standard deviation of 1/sqrt(num_components)

        :return: np.ndarray, shape (num_components, num_features)
            a gaussian matrix of dimension num_components-by-num_features
        '''
        random_generator = RandomState(seed=self.random_state)
        return random_generator.normal(loc=0.0,
                                       scale=1/np.sqrt(self.num_components),
                                       size=(self.num_components, self.num_features))


def johnson_lindenstrauss_bound(n_samples, eps):
    """
    Calculate the Johnson-Lindenstrauss minimum dimensionlaty for distortion eps and n_samples samples

    :param n_samples: int
        Number of samples
    :param eps: float
        Maximum distortion rate
    :return: int
        Minimum number of dimensions that are needed to guarantee with good probability
        an eps-embedding with n_samples
    """
    eps = np.asarray(eps)
    n_samples = np.asarray(n_samples)
    if np.any(eps <= 0) or np.any(eps >= 1.0):
        raise ValueError("eps must be in the interval ]0.0, 1.0[ ")
    if np.any(n_samples <= 0):
        raise ValueError("JL bound is not defined for n_samples <= 0")
    num = 4 * np.log(n_samples)
    den = (eps**2 / 2 - eps**3 / 3)
    return (num / den).astype(np.int)
