import numpy as np


class Node:
    def __init__(self, feature, split_value=None, left_child=None, right_child=None,
                 leaf_value=None):
        self.feature = feature
        self.split_value = split_value
        self.left_child = left_child
        self.right_child = right_child
        self.leaf_value = leaf_value


class DecisionTree:
    def __init__(self,
                 criterion='entropy',
                 minimal_split=2,
                 max_depth=100,
                 num_features_split=None,
                 feature_bagging=False,
                 verbose=False,
                 ):

        self.criterion = criterion
        self.minimal_split = minimal_split
        self.max_depth = max_depth
        self.root_node = None
        self.verbose = verbose
        self.feature_bagging = feature_bagging
        self.num_features_split = num_features_split

    def fit(self, X, y, ):
        self.num_samples, self.num_features = X.shape
        self.root_node = self._grow_decision_tree(X,y)

    def _grow_decision_tree(self, samples, labels, depth=0):
        n_samples, n_features = samples.shape
        unique_labels, labels_count = np.unique(labels, return_counts=True)

        num_classes = unique_labels.shape[0]

        if num_classes == 1 or n_samples <= self.minimal_split or depth >= self.max_depth:
            node_label = unique_labels[np.argmax(labels_count)]
            return Node(leaf_value=node_label)

        split_feature, split_threshold = self._find_best_split_feature(samples, labels)

        left_idx, right_idx = self._split_on_threshold(samples, split_feature, split_threshold)
        left_child = self._grow_decision_tree(samples[left_idx], labels[left_idx], depth + 1)
        right_child = self._grow_decision_tree(samples[right_idx], labels[right_idx], depth + 1)

        return Node(feature=split_feature, split_value=split_threshold,
                    left_child=left_child, right_child=right_child)


    def _find_best_split_feature(self, samples, labels):
        if self.feature_bagging:
            splitting_features_idx = np.random.choice(self.num_features, self.num_features_split, replace=False)
        else:
            splitting_features_idx = range(self.num_features)

        best_information_gain = -1 #TODO:check if there is a different lower bound
        split_feature_idx = None
        split_threshold = None

        for feature_idx in splitting_features_idx:
            feature_values = samples[:, feature_idx]
            split_values = np.unique(feature_values)
            for split_value in split_values:
                information_gain = self._information_gain(feature_values, labels, split_value)
                if information_gain > best_information_gain:
                    best_information_gain = information_gain
                    split_feature_idx = feature_idx
                    split_threshold = split_value

        return split_feature_idx, split_threshold

    def _split_on_threshold(self, samples, split_feature, split_threshold):
        left_split_idx = np.argwhere(samples[:, split_feature] <= split_threshold).flatten()
        right_split_idx = np.argwhere(samples[:, split_feature] > split_threshold).flatten()
        return left_split_idx, right_split_idx

    def _information_gain(self, feature_values, labels, split_value):
        parent_split_quality = self._compute_split_quality(labels)

        # TODO: modify _split_on_threshold
        left_split_idx, right_split_idx = self._split_on_threshold(feature_values, 0, split_value)
        left_split_quality = self._compute_split_quality(labels[left_split_idx])
        right_split_quality = self._compute_split_quality(labels[right_split_idx])
        children_split_quality = left_split_idx.shape[0]/labels.shape[0] * left_split_quality +\
                                 right_split_idx.shape[0]/labels.shape[0] * right_split_quality
        information_gain = parent_split_quality - children_split_quality

        return information_gain


    def _compute_split_quality(self, labels):
        labels_u, labels_c = np.unique(labels, return_counts=True)
        probabilities = labels_c / labels.shape[0]
        # TODO: check computations
        if self.criterion == 'gini':
            return np.sum([class_p * (1-class_p) for class_p in probabilities])
        elif self.criterion == 'entropy':
            return -1 * np.sum([class_p * np.log2(class_p) for class_p in probabilities])

    def predict(self, X):
        y_pred = [self._traverse_decision_tree(x, self.root_node) for x in X[:]]
        return np.array(y_pred)

    # TODO: add print for verbose 
    def _traverse_decision_tree(self, sample, node):
        if node.leaf_value is not None:
            return node.leaf_value
        if sample[node.feature] <= node.split_value:
            return self._traverse_decision_tree(sample, node.left_child)
        else:
            return self._traverse_decision_tree(sample, node.right_child)







