import numpy as np

class SimpleImputer:
    def __init__(self, strategy='mean'):
        self.strategy = strategy
        self.statistics_ = []

    def fit(self, X):
        X = np.array(X, dtype=float)
        self.statistics_ = []

        for col_idx in range(X.shape[1]):
            col = X[:, col_idx]
            valid_vals = col[~np.isnan(col)]

            if self.strategy == 'mean':
                val = np.mean(valid_vals)
            elif self.strategy == 'median':
                val = np.median(valid_vals)
            elif self.strategy == 'most_frequent':
                vals, counts = np.unique(valid_vals, return_counts=True)
                val = vals[np.argmax(counts)]
            else:
                raise ValueError(f"Unknown strategy: {self.strategy}")

            self.statistics_.append(val)

        return self

    def transform(self, X):
        X = np.array(X, dtype=float).copy()

        for col_idx in range(X.shape[1]):
            col = X[:, col_idx]
            nan_mask = np.isnan(col)
            X[nan_mask, col_idx] = self.statistics_[col_idx]

        return X

    def fit_transform(self, X):
        return self.fit(X).transform(X)