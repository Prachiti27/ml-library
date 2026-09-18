import numpy as np

class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None
        
    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        self.scale_[self.scale_ == 0] = 1.0
        return self
    
    def transform(self, X):
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("The scaler instance is not fitted yet. Call 'fit' first")
        
        X = np.asarray(X, dtype=np.float64)
        return (X - self.mean_) / self.scale_
    
    def fit_transform(self, X):
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X):
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("The scaler instance is not fitted yet. Call 'fit' first")
        
        X = np.asarray(X, dtype=np.float64)
        
        return (X * self.scale_) + self.mean_