import numpy as np
from collections import Counter

class KNN:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None
        
    def fit(self, X, y):
        self.X_train = np.asarray(X)
        self.y_train = np.asarray(y)
        
    def _euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x2-x1)**2, axis=1))
    
    def _predict_single(self, x):
        distances = self._euclidean_distance(x, self.X_train)
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = self.y_train[k_indices]
        
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]
    
    def predict(self, X):
        X = np.asarray(X)
        return np.array([self._predict_single(x) for x in X])