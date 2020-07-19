import numpy as np
from scipy import linalg
from .utils import check_data


class FLD:
    def __init__(self, num_components=None, verbose=False):
        self.num_classes = None
        self.num_features = None
        self.num_samples = None
        self.num_components = num_components
        self.max_num_components = None

        self.class_means = None

        self._eigenvects = None
        self.discriminants = None
        self.explained_variance_ratio = None

        self.verbose = verbose

    def fit(self, X, y):
        """

        :param X:
        :param y:
        :return:
        """
        self.num_samples, self.num_features = X.shape

        y_unique, y_count = np.unique(y, return_counts=True)
        self.num_classes = y_unique.shape[0]
        if self.verbose:
            if self.num_classes > 2:
                print("Multiple Discriminant Analysis")
            else:
                print("Fisher's Linear Discriminant")
            print("-"*50)
            print(f"- Data matrix dimensions : {self.num_samples} samples, {self.num_features} features")
            print("-" * 50)
            print(f"- Number of classes : {self.num_classes}")

        self.max_num_components = min(self.num_classes - 1, self.num_features)

        if self.num_components is None:
            self.num_components = self.max_num_components
        elif self.num_components > self.max_num_components:
            raise ValueError("num_components is greater than (num_classes - 1)")

        if self.verbose:
            print(f"The maximum number of discriminants is : {self.max_num_components}.")
            print(f"Number of discriminants selected during initialization is : {self.num_components}")
            print("-"*50)

        # Compute the mean of each class group
        self.class_means = np.zeros(shape=(self.num_classes, self.num_features))
        for i in range(self.num_classes):
            self.class_means[i] = np.mean(X[y == i], axis=0)
        if self.verbose:
            print(f"- Computing class sample means")
            print("-"*50)

        # St : scatter matrix of the whole data matrix
        St = np.cov(X.T, bias=True)
        if self.verbose:
            print("- Computing St: total scatter matrix")
            print(f"St dimensions: {St.shape[0]} x {St.shape[1]}")
            print("-"*30)

        # Sw : within-class scatter matrix
        Sw = np.zeros(shape=(self.num_features, self.num_features))
        for i in range(self.num_classes):
            Sw += 1/X.shape[0] * (((X[y == i]) - self.class_means[i]).T @ (X[y == i] - self.class_means[i]))
            # Alternative Book version
            # Sw += y_count[i] * ((X[y == i] -  class_means[i]).T @ (X[y == i] - class_means[i]))

        if self.verbose:
            print("Computing Sw : within-class scatter matrix")
            print(f"Sw shape : {Sw.shape[0]} x {Sw.shape[1]}")
            print("-"*50)

        # Sb : between-classes scatter matrix
        Sb = St - Sw
        if self.verbose:
            print("Computing Sb : between-classes scatter matrix")
            print(f"Sb shape : {Sb.shape[0]} x {Sb.shape[1]}")
            print("-"*50)

        # To solve the generalized eigenvalue problem scipy provides the function linalg.eigh
        # The actual problem is Sb v = x Sw v
        # https://stackoverflow.com/questions/24752393/solve-generalized-eigenvalue-problem-in-numpy
        eigenvals, eigenvects = linalg.eigh(Sb, Sw)

        self._eigenvects = eigenvects[:, np.argsort(-eigenvals)]
        self.discriminants = self._eigenvects[:, :self.num_components]

        if self.verbose:
            print("Solving generalized eigenvalue problem Sb w = λ Sw w")
            print("-"*50)
            print(f"-Sorting eigenvalues and corresponding eigenvectors")
            print(f"- Taking the {self.num_components} eigenvectors as discriminants")
            print("-"*50)
        return self

    def transform(self, X):
        if self.verbose:
            print("Transform data : ")
            print("-" * 50)
        X_new = X @ self._eigenvects

        if self.verbose:
            print(f"- Computing projection")
            print(f"")
            print(f"\t- Original X dimensions : {X.shape[0]} x {X.shape[1]}")
            print(f"\t- Projection dimensions : {X_new.shape[0]} x {self.num_components}")
            print()
            print("-" * 50)
        return X_new[:, :self.num_components]

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

    def predict(self, X):
        # TODO: implement for classification
        pass
