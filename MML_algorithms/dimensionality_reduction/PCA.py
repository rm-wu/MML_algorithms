import numpy as np
from plotly.offline import iplot
from matplotlib import pyplot as plt

from ..utils import flip_svd, check_data

class PCA:
    def __init__(self, num_components=2, verbose=False):
        '''

        :param num_components: int, default=2
            Number of components to keep
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

        if self.verbose:
            print("Begin PCA")
            print("-" * 50)
            print(f"Data matrix X dimensions : {self.num_samples} samples, {self.num_features} features")
            print("-" * 50)
            print("Centering data matrix")
            print(" - computing X columns means")
            print(" - subtract the mean from the original data matrix")
            print("-" * 50)

        if X.shape[0] > X.shape[1]:
            #TODO: change it to (not shape1 >>> shape0)
            # case A
            A = (X_.T @ X_)
            if self.verbose:
                print("Case A (num_samples > num_features) :")
                print(f"- Compute A = X'X")
                print(f"\t- X'X shape : ({X_.shape[1]} x {X_.shape[0]}) * ({X_.shape[0]} x {X_.shape[1]})")
                print(f"\t            :  {A.shape[0]} x {A.shape[1]}")
                print("-" * 50)

            # A = X'X --> A = USVh --> X'XV = US
            U, S, Vh = np.linalg.svd(A, full_matrices=False)
            # to obtain a deterministic result flip the sign of the eigenvectors which contain the greatest absolute
            # value component with negative sign
            flip_svd(U, Vh)

            self.principal_components = U[:, :self.num_components].T
            singular_values_ = S.copy()

            if self.verbose:
                print(f"- Compute SVD of A = U S Vh")
                print(f"- Dimensionality of the computed matrices")
                print(f"\t- U       (left-singular vectors) : {U.shape[0]} x {U.shape[1]}")

                print(f"\t- diag(S) (singular values)       : {S.shape[0]} x 1 ")
                      #f"\n\t[np.linalg.svd returns only the diagonal of S]")
                print(f"\t- Vh      (right-singular vectors): {Vh.shape[0]} x {Vh.shape[1]}")
                print("-" * 50)
                print(f"- Select the first {self.num_components} left-singular vectors of A with "
                      f"\nthe largest singular values as principal components of X")
                print("-" * 50)

        else:
            # num_features >>> num_samples
            # case B
            B = X_ @ X_.T
            if self.verbose:
                print("Case B (num_features > num_samples) :")
                print(f" - Compute B = XX'")
                print(f" - XX' shape : ({X_.shape[0]} x {X_.shape[1]}) * ({X_.shape[1]} x {X_.shape[0]})")
                print(f"             :  {B.shape[0]} x {B.shape[1]}")
                print("-" * 50)

            # B = XX' --> B = USVh --> XX' = USVh
            U, S, Vh = np.linalg.svd(B)
            flip_svd(U, Vh, U_based=False)

            product = X_.T @ Vh[:self.num_components]  # take the rows
            self.principal_components = product / np.linalg.norm(product, axis=0)
            singular_values_ = S.copy()

            if self.verbose:
                print(f"- Compute SVD of B = U S Vh")
                print(f"- Dimensionality of the computed matrices")
                print(f"\t- U       (left-singular vectors) : {U.shape[0]} x {U.shape[1]}")

                print(f"\t- diag(S) (singular values)       : {S.shape[0]} x 1 ")
                # f"\n\t[np.linalg.svd returns only the diagonal of S]")
                print(f"\t- Vh      (right-singular vectors): {Vh.shape[0]} x {Vh.shape[1]}")
                print("-" * 50)
                print(f"- Select the first {self.num_components} right-singular vectors of B with "
                      f"\nthe largest singular values")
                print(f"- Compute the principal components of X as:"
                      f"\n u_i = (X' v_i)/ ||(X' v_i)|| for i in 1, ... , {self.num_components}")
                print("-" * 50)

        self.singular_values = np.sqrt(singular_values_[:self.num_components])
        self.explained_variance = singular_values_ / (self.num_samples - 1)
        self.explained_variance_ratio = self.explained_variance / self.explained_variance.sum()

        if self.verbose:
            print(f"- Summary of SVD results: ")
            print()
            print(f"- Singular values:")
            for i in range(singular_values_.shape[0]):
                print(f"\t{i+1} - {np.sqrt(singular_values_[i])}")
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
        X_ = X - self.means
        scores_ = X_ @ self.principal_components[:self.num_components].T
        if self.verbose:
            print("Transform data : ")
            print("-" * 50)
            print(f"- Centering data matrix ")
            print(f"- Computing projection")
            print(f"")
            print(f"\t- Original X dimensions : {X.shape[0]} x {X.shape[1]}")
            print(f"\t- Projection dimensions : {scores_.shape[0]} x {scores_.shape[1]}")
            print()
            print("-" * 50)
        return scores_

    #def biplot(self):
    #   pass

    def plot_explained_variance(self, interactive=True):
        if interactive:
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
        else:
            return
