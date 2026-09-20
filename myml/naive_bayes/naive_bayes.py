import numpy as np

class NaiveBayes:
    def __init__(self):
        self.classes = None
        self.mean = {}
        self.var = {}
        self.priors = {}
        
    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        
        num_samples, num_features = X.shape
        
        self.classes = np.unique(y)
        
        for c in self.classes:
            X_c = X[y == c]
            self.mean[c] = np.mean(X_c, axis=0)
            self.var[c] = np.var(X_c, axis=0) + 1e-9
            self.priors[c] = X_c.shape[0] / float(num_samples)
            
    def _pdf(self, class_idx, x):
        mean = self.mean[class_idx]
        var = self.var[class_idx]
        numerator = np.exp(-((x-mean)**2) / (2*var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator
    
    def _predict_single(self, x):
        posteriors = []
        for c in self.classes:
            prior = np.log(self.priors[c])
            conditional = np.sum(np.log(self._pdf(c,x) + 1e-15))
            posterior = prior + conditional
            posteriors.append((posterior, c))
            
        return max(posteriors, key=lambda item: item[0])[1]
    
    def predict(self, X):
        X = np.asarray(X)
        return np.array([self._predict_single(x) for x in X])