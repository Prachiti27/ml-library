import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.lr = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None
        self.losses = []
        
    def fit(self, X, y):
        num_samples, num_features = X.shape
        
        self.weights = np.zeros(num_features)
        self.bias = 0
        
        for _ in range(self.num_iterations):
            z = np.clip(np.dot(X, self.weights) + self.bias, -500, 500)
            y_hat = 1 / (1 + np.exp(-z))
            
            epsilon = 1e-15
            y_hat_safe = np.clip(y_hat, epsilon, 1 - epsilon)
            loss = - np.mean(y*np.log(y_hat_safe) + (1-y)*np.log(1 - y_hat_safe))
            self.losses.append(loss)
            
            dw = (1/num_samples) * np.dot(X.T, (y_hat - y))
            db = (1/num_samples) * np.sum(y_hat - y)
            
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            
    def predict(self, X, threshold=0.5):
        z = np.clip(np.dot(X, self.weights) + self.bias, -500, 500)
        probabilities = 1 / (1 + np.exp(-z))
        return (probabilities >= threshold).astype(int)
        