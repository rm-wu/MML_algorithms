import numpy as np
from plotly.offline import iplot
#from scipy import linalg
from ..utils import flip_svd, check_data

# Data must be normalized before applying PCA

class PCA:
    def __init__(self, num_components=None, verbose=False):
        '''
        :param num_components:
        :param verbose:
        '''
        if num_components <= 0:
            raise ValueError("num_components cannot be less than or equal 0")
        self.num_components = 2 if not None else num_components
        self.num_samples = None
        self.num_features = None

        self.explained_variance = None
        self.explained_variance_ratio = None

        self.verbose = verbose
        self.means = None

        self.principal_components = None
        self.singular_values = None

    def fit(self, X):
        '''

        :param X:
        :return:
        '''

        check_data(X)

        self.num_samples, self.num_features = X.shape
        self.means = X.mean(axis=0)
        X_ = X - self.means
        singular_values_ = None

        if X.shape[0] > X.shape[1]:
            #TODO: change it to (not shape1 >>> shape0)

            # case A
            A = (X_.T @ X_)  # / (X.shape[0] - 1)
            # A = X'X --> A = USVh --> X'XV = US
            U, S, Vh = np.linalg.svd(A, full_matrices=False)  # np.linalg.svd(A)

            # to obtain a deterministic result flip the sign of the eigenvectors which contain the greatest absolute
            # value component with negative sign
            flip_svd(U, Vh)

            self.principal_components = U[:, :self.num_components].T
            singular_values_ = S.copy()

            if self.verbose:
                print("Case A (m > d)")
                print(f"A shape = {A.shape[0]} x {A.shape[1]}")

        else: # num_features >>> num_samples
            # case B
            B = X_ @ X_.T
            # B = XX' --> B = USVh --> XX' = USVh
            U, S, Vh = np.linalg.svd(B)
            flip_svd(U, Vh, U_based=False)

            product = X_.T @ Vh[:self.num_components]  # take the rows
            self.principal_components = product / np.linalg.norm(product, axis=0)
            singular_values_ = S

        self.singular_values = np.sqrt(singular_values_[:self.num_components])
        self.explained_variance = singular_values_ / (self.num_samples - 1)
        self.explained_variance_ratio = self.explained_variance / self.explained_variance.sum()
        return self

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def transform(self, X):
        check_data(X)
        X_ = X - self.means
        loadings_ = X_ @ self.principal_components[:self.num_components].T
        return loadings_

    def biplot(self):

        pass

    def plot_explained_variance(self, interactive=True):
        if interactive:
            trace1 = dict(
                type='bar',
                x=['PC %s' % i for i in range(1, self.num_features + 1)],
                                              # self.principal_components.shape[0] + 1)],
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
        else:
            # TODO: add visualization with matplot
            return
