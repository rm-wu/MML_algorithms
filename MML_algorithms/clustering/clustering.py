import random
import numpy as np
import matplotlib.pyplot as plt
from ..utils import euclidean_distance

class KMeans:
    def __init__(self, n_clusters, max_iter=100, tolerance=0.01):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centroids = None
        self.labels = None
        self.tolerance = tolerance
        self._distances = None

    @staticmethod
    def _init_centroids(X, K):
        '''

        :param X:
        :param K:
        :return:
        '''
        # generate K random indexes
        centroids_idx = np.random.permutation(X.shape[0])[:K]
        centroids = X[centroids_idx]
        return centroids

    def fit(self, X, plot_clusters=False, plot_step=5, verbose=False, final_plot=False):
        '''

        :param X:
        :param plot_clusters:
        :param plot_step:
        :param verbose:
        :param final_plot:
        :return:
        '''
        self.centroids = self._init_centroids(X, self.n_clusters)
        max_change = float('inf')
        iteration = 0

        while max_change > self.tolerance and iteration < self.max_iter:
            # assign at each centroids its points
            self.labels, self._distances = self._update_labels(X)

            if plot_clusters and (iteration % plot_step) == 0:
                self._plot_clusters(X)

            old_centroids = self.centroids.copy()
            self._update_centroids(X)

            max_change = self._calculate_change(old_centroids)
            if verbose:
                print(iteration, max_change)

            iteration += 1

        if plot_clusters:
            self._plot_clusters(X)

        return

    def _update_labels(self, X):
        '''

        :param X:
        :return:
        '''
        centers = self.centroids.reshape(self.n_clusters, 1, X.shape[1])
        distances = np.sqrt(np.sum((X-centers)**2, axis=-1))
        labels = np.argmin(distances, axis=0)
        distances = np.min(distances, axis=0)
        return labels, distances

    def _update_centroids(self, X):
        '''

        :param X:
        :return:
        '''
        # let empty clusters be updated with the most distant point
        unique, counts = np.unique(self.labels, return_counts=True)
        dict_cont = dict(zip(unique, counts))
        for i in range(self.n_clusters):
            if i not in dict_cont or dict_cont[i]==1:
                idx_new_centroid = np.argmax(self._distances)
                self.centroids[i] = X[idx_new_centroid, :]
                self._distances[idx_new_centroid] = 0
                self.labels[idx_new_centroid] = i

        # update centroids
        for i in dict_cont.keys():
            cluster_points = X[self.labels == i, :]
            if cluster_points.shape[0] == 0:
                print("ERROR!!!")
            else:
                self.centroids[i] = np.mean(cluster_points, axis=0)
        return

    def _calculate_change(self, old_centroids):
        '''

        :param old_centroids:
        :return:
        '''
        changes = np.sqrt(np.sum((self.centroids - old_centroids) ** 2, axis=-1))
        max_change = np.max(changes)
        return max_change

    def _plot_clusters(self, X):
        fig, ax = plt.subplots()
        ax.scatter(X[:, 0], X[:, 1], c=self.labels)
        ax.scatter(self.centroids[:, 0], self.centroids[:, 1], c='red', marker='*', s=100)
        plt.show()

def silhouette_samples(X, labels):
    """Evaluate the silhouette for each point and return them as a numpy array.

    :param X: input data points, array, shape = (N,C).
    :param labels: the list of cluster labels, shape = N.
    :return: silhouette : array, shape = N
    """
    unique, counts = np.unique(labels, return_counts=True)
    np_counts = np.array(counts)
    fact_a = 1/(np_counts - 1)
    fact_b = 1/np_counts

    a = np.zeros(shape=(X.shape[0], ))
    b = np.zeros(shape=(X.shape[0], ))
    for i in range(X.shape[0]):
        a[i] = fact_a[i] * np.sum(euclidean_distance(X[i, :], X[labels == labels[i], :]))

    for i in range(X.shape[0]):
        min = float('inf')
        for j in unique:
            if labels[i]!=j:
                b_j = fact_b[j] * np.sum(euclidean_distance(X[i, :], X[labels == j, :]))
                if b_j < min:
                    min = b_j
        b[i] = min
    max_a_b = np.max(np.array([a, b]), axis=0)
    silhouette = (b - a)/max_a_b
    return silhouette

def silhouette_score(X, labels):
    '''
    Evaluate the silhouette for each point and return the mean.

    :param X: input data points, array, shape = (N,C). :param labels: the list of cluster labels, shape = N.
    :return: silhouette : float
    '''
    silhouette = silhouette_samples(X, labels)
    return np.mean(silhouette)

