import numpy as np
from ..decision_tree.decision_tree_classifier import DecisionTreeClassifier

class RandomForestClassifier:
    def __init__(self, n_estimators=100, max_depth=None, max_features="sqrt",random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.random_state = random_state
        self.trees = []
        
    def _bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        return X[indices], y[indices]
    
    def fit(self, X, y):
        if self.random_state is not None:
            np.random.seed(self.random_state)
            
        self.trees = []
        for _ in range(self.n_estimators):
            tree = DecisionTreeClassifier(max_depth=self.max_depth, max_features=self.max_features)
            X_sample, y_sample = self._bootstrap_sample(X, y)
            
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
            
    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = tree_preds.T
        predictions = [np.bincount(sample_preds).argmax() for sample_preds in tree_preds]
        return np.array(predictions)