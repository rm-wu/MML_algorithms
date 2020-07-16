import numpy as np
from ..utils import check_data


def linear_kernel(x1, x2, b=0.0):
    return (x1 @ x2.T) + b


def rbf_kernel(x1, x2, gamma=1):
    if x1.ndim == 1 and x2.ndim == 1:
        # 1) basic case 2 arrays
        result = np.exp(- (np.linalg.norm(x1 - x2, 2))**2 * gamma) #/ (2 * sigma ** 2))
    elif (x1.ndim > 1 and x2.ndim == 1) or (x1.ndim == 1 and x2.ndim > 1):
        # 2) one array and 2-D array:
        # if one of the two elements is a matrix its additional dimension is broadcasted with numpy
        # to take the norm of the resulting array
        result = np.exp(- (np.linalg.norm(x1 - x2, 2, axis=1) ** 2) * gamma)#/ (2 * sigma ** 2))
    elif x1.ndim > 1 and x2.ndim > 1:
        # 3) two 2-D arrays:
        # x1[:, np.newaxis] - x2[np.newaxis, :] => it is equal to the difference between each couple of rows
        # (i, j)-th element of the result contains the difference between x1[i] and x2[j]
        # linalg.norm on the 2nd axis takes the norm of the resulting array
        result = np.exp(- (np.linalg.norm(x1[:, np.newaxis] - x2[np.newaxis, :], axis=2) ** 2) * gamma) # / (2 * sigma ** 2))
    return result


#def poly_kernel(x1, x2, c=0.0, degree=2):
#    return (x1 @ x2.T + c)**degree


class SVM:
    def __init__(self, solver='SGD', kernel='linear', degree=2, C=0, gamma=1,
                 max_iterations=10000, step_size=1e-3, tol=1e-2, eps=1e-2,
                 verbose=False):
        # Initialize solver type and hyperparameters
        if solver not in ['SGD', 'SMO']:
            raise ValueError("Parameter 'solver' can only take these values:\n"
                             "\t'SGD' : to use Stochastic Gradient Descent \n"
                             "\t'SMO' : to use Sequential Minimum Optimization")
        self.solver = solver
        if solver == 'SMO':
            self.tol = tol
            self.eps = eps
            self._errors = None
            self.C = C
        else:
            self.step_size = step_size
            self.C = 1/C
        self.max_iterations = int(max_iterations)

        # Initialize kernel type
        if kernel == 'linear':
            self.kernel = linear_kernel
        elif kernel == 'rbf':
            self.gamma = gamma
            self.kernel = lambda x1, x2: rbf_kernel(x1, x2, gamma=self.gamma)
        #elif kernel == 'poly':
        #    self.degree = degree
        #    self.kernel = lambda x1, x2: poly_kernel(x1, x2, degree=self.degree)
        else:
            raise ValueError("Parameter 'kernel' can only take these values:\n"
                             "\t'linear' : \n"
                             "\t'poly'   : \n"
                             "\t'rbf'    : \n")
        #self.C = C
        self.W = None
        self.b = 0.0
        self.alphas = None
        self.X_train = None

    def fit(self, X, y):
        if self.solver == 'SGD':
            print("fit model using SGD")
            self._fit_SGD(X, y)
        elif self.solver == 'SMO':
            self._fit_SMO(X, y)
        return self

    def _fit_SGD(self, X, y):
        self.X_train, self.y_train = X, y
        self.W = np.zeros(self.X_train.shape[1])
        iter_per_epoch = X.shape[0]
        num_epochs = self.max_iterations // iter_per_epoch
        for epoch in range(num_epochs):
            shuffled_idx = np.random.permutation(X.shape[0])
            for i in shuffled_idx:
                if self.y_train[i] * (np.dot(self.W, X[i]) - self.b) <= 1:
                    #self.W = (1 - self.step_size) * self.W + self.step_size * self.C * X[i, :]
                    #self.b = self.b - self.step_size * self.y_train[i]
                    update_value_of_w = self.C * self.W - self.y_train[i] * self.X_train[i]
                    update_value_of_b = self.y_train[i]
                else:
                    update_value_of_w = self.C * self.W
                    update_value_of_b = 0
                    #self.W = (1 - self.step_size) * self.W
                    # weights and bias updates
                self.W -= self.step_size * update_value_of_w
                self.b -= self.step_size * update_value_of_b
        return

    def _objective_function(self, X, y, alphas=None):
        if alphas is None:
            alphas = self.alphas
        obj_fun = np.sum(alphas) - 0.5 * np.sum((y[:, np.newaxis] * y[np.newaxis, :])
                                                 * self.kernel(X, X)
                                                 * (alphas[:, np.newaxis] * alphas[np.newaxis, :]))
        return obj_fun

    def decision_function(self, X_test):
        if self.solver == 'SMO':
            result = (self.alphas * self.y_train) @ self.kernel(self.X_train, X_test) - self.b
        elif self.solver == 'SGD':
            result = np.dot(X_test, self.W) - self.b
        return result

    def _fit_SMO(self, X, y):
        # TODO: check that all the y are or 1 or -1
        check_data(X)

        # Initialization of all the elements needed for the optimization
        self.num_samples, self.num_features = X.shape
        self.X_train = X
        self.y_train = y
        self.alphas = np.zeros(self.num_samples)
        self._errors = self.decision_function(X) - self.y_train

        self._train_SMO()

    def _take_step(self, idx1, idx2):
        # If the alphas selected are the same skip
        if idx1 == idx2:
            return 0

        # Initialize the two alphas and corresponding values
        alpha1 = self.alphas[idx1]
        alpha2 = self.alphas[idx2]
        y1 = self.y_train[idx1]
        y2 = self.y_train[idx2]
        E1 = self._errors[idx1]
        E2 = self._errors[idx2]
        s = y1 * y2

        # Compute L and H, i.e. the bounds on the new possible alpha values
        if y1 != y2:
            L = max(0, alpha2 - alpha1)
            H = min(self.C, self.C + alpha2 - alpha1)
        else: # y1 == y2
            L = max(0, alpha2 + alpha1 - self.C)
            H = min(self.C, alpha2 + alpha1)
        if L == H:
            return 0

        # Compute the kernels for point1 and point2
        # TODO: put kernel implementaiton inside the class
        k11 = self.kernel(self.X_train[idx1], self.X_train[idx1])
        k12 = self.kernel(self.X_train[idx1], self.X_train[idx2])
        k22 = self.kernel(self.X_train[idx2], self.X_train[idx2])
        # Compute second derivative eta
        eta = 2 * k12 - k11 - k22

        if eta < 0:
            # Compute the new value of alpha2 (a2) if eta is negative
            a2 = alpha2 - y2 * (E1 - E2) / eta
            # Clip the value of a2 between the bounds L and H
            if a2 <= L:
                a2 = L
            elif a2 >= H:
                a2 = H
        else:
            # If eta is non-negative, move a2 to the bound with greater objective function value
            alphas_temp = self.alphas.copy()
            # Lobj: objective function obtained with new alpha2 = L
            alphas_temp[idx2] = L
            Lobj = self._objective_function(self.X_train, self.y_train, alphas=alphas_temp)

            # Hobj: objective function obtained with new alpha2 = H
            alphas_temp[idx2] = H
            Hobj = self._objective_function(self.X_train, self.y_train, alphas=alphas_temp)

            if Lobj > (Hobj + self.eps):
                a2 = L
            elif Lobj < (Hobj - self.eps):
                a2 = H
            else:
                a2 = alpha2

        # Push a2 to 0 or C if they are close enough
        if a2 < 1e-8:
            a2 = 0.0
        elif a2 + 1e-8 > self.C:
            a2 = self.C

        # If samples cannot be optimized skip this pair
        if np.abs(a2 - alpha2) < self.eps * (a2 + alpha2 + self.eps):
            return 0

        # Calculate the new value of alpha1 (a1)
        a1 = alpha1 + s * (alpha2 - a2)

        # Update the bias term obtained with the new alphas
        # b1 is valid when a1 is not at bounds
        b1 = E1 + y1 * (a1 - alpha1) * k11 + y2 * (a2 - alpha2) * k12 + self.b
        # b2 is valid when a2 is not at bounds
        b2 = E2 + y1 * (a1 - alpha1) * k12 + y2 * (a2 - alpha2) * k22 + self.b

        if 0.0 < a1 < self.C:
            b_ = b1
        elif 0.0 < a2 < self.C:
            b_ = b2
        else:
            # if both alphas are not at bounds then the interval between b1 and b2
            # are all consistent with the KKT conditions. SMO chooses the bias to
            # be : b_ = (b1 + b2) / 2
            b_ = 0.5 * (b1 + b2)

        self.alphas[idx1] = a1
        self.alphas[idx2] = a2
        # Update bias term


        # Update self._errors
        # Optimized alphas
        for idx, alpha in zip([idx1, idx2], [a1, a2]):
            if 0.0 < alpha < self.C:
                self._errors[idx] = 0.0
        # Non-optimized alphas
        non_opt_idx = [i for i in range(self.num_samples) if (i != idx1) and (i != idx2)]
        self._errors[non_opt_idx] = self._errors[non_opt_idx] \
                                    + (y1 * (a1 - alpha1) * self.kernel(self.X_train[idx1], self.X_train[non_opt_idx])) \
                                    + (y2 * (a2 - alpha2) * self.kernel(self.X_train[idx2], self.X_train[non_opt_idx])) \
                                    + self.b - b_
        self.b = b_
        return 1

    def _examine_sample(self, idx2):
        y2 = self.y_train[idx2]
        alpha2 = self.alphas[idx2] # Lagrange multiplier for idx_2
        E2 = self._errors[idx2]
        r2 = E2 * y2
        #print(f"examine_example {idx2}", y2, alpha2, E2, r2)

        if (r2 < -self.tol and alpha2 < self.C) or (r2 > self.tol and alpha2 > 0):
            # if number of non-zero alphas > 1 and non-C alphas > 1
            if len(self.alphas[(self.alphas != 0) & (self.alphas != self.C)]) > 1:
                # second choice heuristic
                if self._errors[idx2] > 0:
                    idx1 = np.argmin(self._errors)
                else: # self._errors[idx2] <= 0
                    idx1 = np.argmax(self._errors)

                step_result = self._take_step(idx1, idx2)
                if step_result:
                    return 1

            # Loop through all non-zero and non-C alphas, starting at a random point
            # j = np.random.choice(np.arange(self.num_samples)) # starting index
            # alphas_mask = (self.alphas != 0) & (self.alphas != self.C)
            # for _ in range(self.num_samples):
            #     if alphas_mask[j]:
            #         step_result = self._take_step(j, idx2)
            #         if step_result:
            #             return 1
            #     j = (j + 1) % self.num_samples
            for idx1 in np.roll(np.where((self.alphas != 0) & (self.alphas != self.C))[0],
                              np.random.choice(np.arange(self.num_samples))):
                step_result = self._take_step(idx1, idx2)
                if step_result:
                    return 1

            # # Loop through all idx1, starting at a random point
            # j = np.random.choice(np.arange(self.num_samples))  # starting index
            # for _ in range(self.num_samples):
            #     if self._take_step(j, idx2):
            #         return 1
            #     j = (j + 1) % self.num_samples
            for idx1 in np.roll(np.arange(self.num_samples),
                                np.random.choice(np.arange(self.num_samples))):
                #print("loop2", idx1)
                step_result = self._take_step(idx1, idx2)
                if step_result:
                    return 1

        return 0

    def _train_SMO(self):
        num_changed = 0
        examine_all = True
        # n_iter = 0
        while num_changed > 0 or examine_all:
            num_changed = 0
            if examine_all:
                #print(n_iter, "examine_all")
                # loop over all the training samples
                for i in range(self.num_samples):
                    examine_result = self._examine_sample(i)
                    num_changed += examine_result
            else:
                #print(n_iter, "num_changed")
                for idx1 in np.where((self.alphas != 0) & (self.alphas != self.C))[0]:
                    examine_result = self._examine_sample(idx1)
                    num_changed += examine_result
                # loop over all the alphas != 0 and != C
                # indexes = np.argwhere((self.alphas != 0.0) & (self.alphas != self.C))
                # if len(indexes) > 0:
                #     for idx in indexes[0]:
                #         examine_result = self._examine_sample(idx)
                #         num_changed += examine_result
            #print(n_iter, num_changed)
            if examine_all:
                examine_all = False
            elif num_changed == 0:
                examine_all = True
            # n_iter += 1
        return

    def predict(self, X):
        prediction = self.decision_function(X)
        #print(np.sign(prediction))
        return np.sign(prediction)


