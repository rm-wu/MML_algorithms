import numpy as np
from plotly.offline import iplot
import plotly.offline
from matplotlib import pyplot as plt

from ..utils import flip_svd, check_data

class PCA:
    def __init__(self, num_components=2, verbose=False):
        '''

        :param num_components: int, default=2
            Number of components to keep
        :param centering_data: bool, default=True
            Centers the data matrix before performing PCA
        :param verbose: bool, default=False
            If True the class will print brief information about the operations performed by the class
        '''
        if num_components <= 0:
            raise ValueError("num_components cannot be less than or equal 0")
        self.num_components = num_components
        self.num_samples = None
        self.num_features = None

        self.explained_variance = None
        self.explained_variance_ratio = None

        self.verbose = verbose
        self.means = None

        self.principal_components = None
        #self.singular_values = None


    def fit(self, X):
        '''

        :param X:
        :return:
        '''
        check_data(X)
        self.num_samples, self.num_features = X.shape
        if self.verbose:
            print("Begin PCA")
            print("-" * 50)
            print(f"Data matrix dimensions : {self.num_samples} samples, {self.num_features} features")
            print("-" * 50)

        self.means = X.mean(axis=0)
        X_ = X - self.means
        if self.verbose:
            print("Centering data matrix")
            print(" - computing X columns means")
            print(" - subtract the mean from the original data matrix")
            print("-" * 50)

        singular_values_ = None

        U, S, Vh = np.linalg.svd(X_, full_matrices=False)
        # to obtain a deterministic result flip the sign of the singular vectors which contain the greatest absolute
        # value component with negative sign
        flip_svd(U, Vh)

        #self.principal_components = U[:, :self.num_components].T
        self.principal_components = Vh[:self.num_components]
        singular_values_ = S.copy()

        if self.verbose:
            print(f"- Compute SVD:  X = U S Vt")
            print(f"- Dimensionality of the computed matrices")
            print(f"\t- U       (left-singular vectors) : {U.shape[0]} x {U.shape[1]}")

            print(f"\t- S       (singular values)       : {S.shape[0]} x {S.shape[0]}")
                  #f"\n\t[np.linalg.svd returns only the diagonal of S]")
            print(f"\t- V       (right-singular vectors): {Vh.shape[1]} x {Vh.shape[0]}")
            print("-" * 50)
            print(f"- Select the first {self.num_components} right-singular vectors of X with "
                  f"\nthe largest singular values as principal components")
            print("-" * 50)

        #self.eigen_values = (singular_values_ ** 2)/(self.num_samples - 1)  #np.sqrt(singular_values_[:self.num_components])
        self.explained_variance = (singular_values_ ** 2)/(self.num_samples - 1)
        self.explained_variance_ratio = self.explained_variance / self.explained_variance.sum()

        if self.verbose:
            print(f"- Summary of SVD results: ")
            print()
            print(f"- Eigen Values of Cov(X):")
            for i in range(self.explained_variance.shape[0]):
                print(f"\t{i+1} - {self.explained_variance[i]}")
            print()
            print(f"- Explained variance by each principal component")
            for i in range(self.explained_variance_ratio.shape[0]):
                print(f"\tPC{i+1} :  {self.explained_variance_ratio[i]}")
            print("-"*50)

        return self

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def transform(self, X):
        check_data(X)

        if self.verbose:
            print("Transform data : ")
            print("-" * 50)

        if self.centering_data:
            X_ = X - self.means
            if self.verbose:
                print("Centering data matrix")
                print("-" * 50)
        else:
            X_ = X.copy()
        scores_ = X_ @ self.principal_components[:self.num_components].T

        if self.verbose:
            print(f"- Computing projection")
            print(f"")
            print(f"\t- Original X dimensions : {X.shape[0]} x {X.shape[1]}")
            print(f"\t- Projection dimensions : {scores_.shape[0]} x {scores_.shape[1]}")
            print()
            print("-" * 50)
        return scores_

    #def biplot(self):
    #   pass

    def plot_explained_variance(self):
        trace1 = dict(
            type='bar',
            x=['PC %s' % i for i in range(1, self.num_features + 1)],
            y=self.explained_variance_ratio,
            name='Individual'
        )

        trace2 = dict(
            type='scatter',
            x=['PC %s' % i for i in range(1, self.num_features + 1)],
            y=np.cumsum(self.explained_variance_ratio),
            name='Cumulative'
        )

        data = [trace1, trace2]

        layout = dict(
            title='Explained variance by different principal components',
            yaxis=dict(
                title='Explained variance in percent'
            ),
            annotations=list([
                dict(
                    x=1.16,
                    y=1.05,
                    xref='paper',
                    yref='paper',
                    text='Explained Variance',
                    showarrow=False,
                )
            ])
        )

        fig = dict(data=data, layout=layout)
        iplot(fig, filename='selecting-principal-components')
        #plotly.offline.init_notebook_mode(connected=True)
