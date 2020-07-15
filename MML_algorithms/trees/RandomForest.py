import numpy as np
from .DecisionTree import DecisionTree


def bootstrap_data(X, y):
    n_samples = X.shape[0]
    random_idx = np.random.choice(n_samples, n_samples, replace=True) # sample with replacement
    return X[random_idx], y[random_idx]

#TODO: max_depth=None??
class RandomForest:
    def __init__(self,
                 num_trees=100,
                 criterion='entropy',
                 max_depth=10,
                 min_samples_split=2,
                 max_features='sqrt',
                 verbose=False
                 ):
        self.num_trees = num_trees
        if criterion in ['gini', 'entropy']:
            self.criterion = criterion
        else:
            #TODO: raise vaulue
            raise ValueError("")
        self.max_depth = max_depth
        self.min_sample_split = min_samples_split
        #TODO: leave only the possibility for sqrt and log2 e al max un int?
        if max_features in ['sqrt', 'log2']:
            self.max_features = max_features
        elif isinstance(max_features, int):
            self.max_features = max_features
        else:
            raise ValueError("max_features must be 'sqrt', 'log2' or an int value")
        self.trees = None
        self.verbose = verbose

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.num_trees):
            tree = DecisionTree(criterion=self.criterion,
                                min_samples_split=self.min_sample_split,
                                max_depth=self.max_depth,
                                max_features=self.max_features,
                                feature_bagging=True)
            bootstrap_X, bootstrap_y = bootstrap_data(X, y)
            tree.fit(bootstrap_X, bootstrap_y)
            self.trees.append(tree)
        return self

    def predict(self, X):
        y_pred = np.array([tree.predict(X) for tree in self.trees])
        print(y_pred.shape)
        y_pred = np.array([self._majority_vote(y_pred[:, i]) for i in range(X.shape[0])])
        print(y_pred.shape)
        return y_pred

    def _majority_vote(self, y_pred):
        y_pred_u, y_pred_c = np.unique(y_pred, return_counts=True)
        #label = np.argmax(y_pred_c)
        #print(np.argmax(y_pred_c))
        #print(y_pred_u, y_pred_c)
        #print(y_pred_u[label])
        return y_pred_u[np.argmax(y_pred_c)]