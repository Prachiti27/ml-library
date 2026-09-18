import numpy as np

class LinearRegression:
    def __init__(self, method="ols", lr=0.01, epochs=1000):
        self.method = method
        self.lr = lr
        self.epochs = epochs
        
        self.coef_ = None
        self.intercept_ = None
        
    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1, 1)
        
        n_samples, n_features = X.shape
        
        if self.method == "ols":
            X_b = np.c_[np.ones((n_samples, 1)), X]
            weights_b = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
            self.intercept_ = weights_b[0][0]
            self.coef_ = weights_b[1:].ravel()
        elif self.method == "gd":
            weights = np.zeros((n_features, 1))
            bias = 0.0
            
            for _ in range(self.epochs):
                y_pred = (X @ weights) + bias
                error = y_pred - y
                dw = (2 / n_samples) * (X.T @ error)
                db = (2 / n_samples) * np.sum(error)
                
                weights -= self.lr * dw
                bias -= self.lr * db
                
            self.coef_ = weights.ravel()
            self.intercept_ = float(bias)
        else:
            raise ValueError("Method must be 'ols' or 'gd")
        
    def predict(self, X):
        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("The model is not fitted yet. Call 'fit first.")
        
        X = np.asarray(X, dtype=np.float64)
        return (X @ self.coef_) + self.intercept_