import numpy as np

class Node:
    def __init__(self, feature=None, threshold=None,left=None,right=None,*,value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        
    def is_leaf(self):
        return self.value is not None
    
class DecisionTreeClassifier:
    def __init__(self, max_depth=10, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        
    def fit(self, X, y):
        self.n_classes = len(np.unique(y))
        self.root = self._build_tree(X, y, depth=0)
        
    def _build_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))
        
        if(depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)
        
        best_feature, best_threshold = self._best_split(X, y, n_features)
        
        if best_feature is None:
            return Node(value=self._most_common_label(y))
        
        left_idxs = X[:, best_feature] <= best_threshold
        right_idxs = ~left_idxs
        
        left_child = self._build_tree(X[left_idxs], y[left_idxs], depth+1)
        right_child = self._build_tree(X[right_idxs], y[right_idxs], depth+1)
        
        return Node(feature=best_feature, threshold=best_threshold, left=left_child, right=right_child)
    
    def _best_split(self, X, y, n_features):
        best_gain = -1.0
        best_features, best_threshold = None, None
        
        current_gini = self._gini_impurity(y)
        
        for feature_idx in range(n_features):
            X_column = X[:, feature_idx]
            thresholds = np.unique(X_column)
            
            for threshold in thresholds:
                gain = self._information_gain(y, X_column, threshold, current_gini)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
        return best_feature, best_threshold
    
    def _information_gain(self, y, X_column, threshold, parent_gini):
        left_idxs = X_column <= threshold
        right_idxs = ~left_idxs
        
        if len(y[left_idxs]) == 0 or len(y[right_idxs]) == 0:
            return 0
        
        n = len(y)
        n_left, n_right = len(y[left_idxs]), len(y[right_idxs])
        gini_left = self._gini_impurity(y[left_idxs])
        gini_right = self._gini_impurity(y[right_idxs])
        
        weighted_gini = (n_left/n) * gini_left + (n_right/n) * gini_right
        return parent_gini - weighted_gini
    
    def _gini_impurity(self, y):
        counts = np.bincount(y)
        probabilities = counts / len(y)
        return 1.0 - np.sum(probabilities**2)
    
    def _most_common_label(self, y):
        return np.bincount(y).argmax()
    
    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])
    
    def _traverse_tree(self, x, node):
        if node.is_leaf():
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
        