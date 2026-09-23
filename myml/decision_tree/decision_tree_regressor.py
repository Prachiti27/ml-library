import numpy as np

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        
    def is_leaf(self):
        return self.value is not None
    
class DecisionTreeRegressor:
    def __init__(self, max_depth=10, min_samples_split=2,max_features="sqrt"):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.root = None
        
    def fit(self, X, y):
        self.root = self._build_tree(X, y, depth=0)
        
    def _build_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        
        if(depth >= self.max_depth or n_samples < self.min_samples_split or np.var(y) == 0):
            leaf_value = np.mean(y)
            return Node(value=leaf_value)
        
        best_feature, best_threshold = self._best_split(X, y, n_features)
        
        if best_feature is None:
            return Node(value=np.mean(y))
        
        left_idxs = X[:, best_feature] <= best_threshold
        right_idxs = ~left_idxs
        
        left_child = self._build_tree(X[left_idxs], y[left_idxs], depth+1)
        right_child = self._build_tree(X[right_idxs], y[right_idxs], depth+1)
        
        return Node(feature=best_feature, threshold=best_threshold, left=left_child, right=right_child)
    
    def _best_split(self, X, y, n_features):
        best_variance_reduction = 0.0  # Require positive reduction
        best_feature, best_threshold = None, None

        current_variance = np.var(y)

        for feature_idx in range(n_features):
            X_column = X[:, feature_idx]
            thresholds = np.unique(X_column)

            for threshold in thresholds:
                # FIX 1: Pass 'threshold' argument
                var_red = self._variance_reduction(
                    y, X_column, threshold, current_variance
                )

                if var_red > best_variance_reduction:
                    best_variance_reduction = var_red
                    best_feature = feature_idx
                    best_threshold = threshold

        return best_feature, best_threshold

    def _variance_reduction(self, y, X_column, threshold, parent_variance):
        left_idxs = X_column <= threshold
        right_idxs = ~left_idxs

        if len(y[left_idxs]) == 0 or len(y[right_idxs]) == 0:
            return 0

        n = len(y)
        n_left, n_right = len(y[left_idxs]), len(y[right_idxs])
        var_left = np.var(y[left_idxs])

        var_right = np.var(y[right_idxs])

        weighted_variance = (n_left / n) * var_left + (n_right / n) * var_right
        return parent_variance - weighted_variance
    
    def _traverse_tree(self, x, node):
        if node.is_leaf():
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
    
    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])