import numpy as np
from matplotlib import pyplot as plt
from scipy import linalg
import seaborn as sns
import pandas as pd

from .utils import  check_data
from sklearn.preprocessing import StandardScaler #TODO: replace
# TODO: Bi-plot visualization https://stats.stackexchange.com/questions/82497/can-the-scaling-values-in-a-linear-discriminant-analysis-lda-be-used-to-plot-e
class LDA:
    # https://github.com/bot13956/linear-discriminant-analysis-iris-dataset/blob/master/LDA_irisdataset.py
    def __init__(self, num_components=None, verbose=False):
        self.num_classes = None
        self.num_features = None
        self.num_samples = None
        self.num_components = num_components

        # TODO: refactor with correct name
        self.class_X_bar = None
        self.X_bar = None

        self.eigenvects = None
        self.explained_variance_ratio = None

        self.verbose = verbose

    def fit(self, X, y):
        '''

        :param X:
        :param y:
        :return:
        '''
        self.num_samples, self.num_features = X.shape
        print(f"X shape : {self.num_samples} x {self.num_features} ")

        y_unique, y_count = np.unique(y, return_counts=True)
        self.num_classes = y_unique.shape[0]

        self.max_num_components = min(self.num_classes - 1, self.num_features)
        # TODO: address problem num_components not initialized == None
        if self.num_components is None:
            self.num_components = self.max_num_components
        elif self.max_num_components > self.max_num_components:
            #TODO: raise error
            return

        if self.verbose:
            print(f"The maximum number of components is {self.max_num_components}.")
            print(f"The number of components selected during initialization is : {self.num_components}")

        # Compute the mean of each class group
        self.class_means = np.zeros(shape=(self.num_classes, self.num_features))
        for i in range(self.num_classes):
            self.class_means[i] = np.mean(X[y == i], axis=0)
        if self.verbose:
            print("Computing class sample means: ")
            print(self.class_means)
            print()

        # St : scatter matrix of the whole data matrix
        St = np.cov(X.T, bias=True)
        if self.verbose:
            print("Computing St: total scatter matrix")
            print(St)
            print("-"*30)

        # Sw : within-class scatter matrix
        Sw = np.zeros(shape=(self.num_features, self.num_features))
        for i in range(self.num_classes):
            Sw += 1/X.shape[0] * (((X[y == i]) - self.class_means[i]).T @ (X[y == i] - self.class_means[i]))
            # Book version
            # Sw += y_count[i] * ((X[y == i] -  class_means[i]).T @ (X[y == i] - class_means[i]))

        if self.verbose:
            print("Computing Sw : within-class scatter matrix")
            print(f"{Sw}")
            print("-"*30)

        # Sb : between-classes scatter matrix
        Sb = St - Sw
        if self.verbose:
            print("Computing Sb : between-classes scatter matrix")
            print(f"{Sb}")
            print("-"*30)

        # To solve the generalized eigenvalue problem scipy provides the function linalg.eigh
        # The actual problem is Sb v = x Sw v
        # https://stackoverflow.com/questions/24752393/solve-generalized-eigenvalue-problem-in-numpy
        eigenvals, eigenvects = linalg.eigh(Sb, Sw)
        print("eigenvect", eigenvects.shape)
        print("eigenvals", eigenvals.shape)

        self.eigenvects = eigenvects[:, np.argsort(-eigenvals)]
        eigenvals = np.sort(eigenvals)[::-1]
        self.explained_variance_ratio = eigenvals / np.sum(eigenvals)

        if self.verbose:
            print("Solving generalized eigenvalue problem Sb v = x Sw v")
            print("(where x is the eigenvalue and v the eigenvector)")

            print("Sorted eigenvalues :")
            for eigval in eigenvals:
                print(eigval)
            print()
            print("Sorted eigenvectors :")
            for eigvect in self.eigenvects:
                print(eigvect)
            print()
            print("Explained variance ratio: ")
            for i, evc in enumerate(self.explained_variance_ratio):
                print(f"LDA{i + 1} : {evc}")

        return self

    def transform(self, X):
        if self.verbose:
            print(X.shape)
            print(self.eigenvects.shape)

        X_new = X @ self.eigenvects
        return X_new[:, :self.num_components]

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

    def predict(self, X):
        # TODO: implement for classification
        return