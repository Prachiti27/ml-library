import numpy as np


class SVM:
    def __init__(self, C=1.0, kernel="rbf", gamma="scale", n_iters=1000, lr=0.001):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.n_iters = n_iters
        self.lr = lr
        self.X_train = None
        self.y_train = None
        self.alpha = None
        self.b = 0.0
        self.classes_ = None

    def _kernel_matrix(self, X1, X2):
        if self.kernel == "linear":
            return np.dot(X1, X2.T)

        elif self.kernel == "rbf":
            if self.gamma == "scale":
                gamma_val = 1.0 / (X1.shape[1] * X1.var()) if X1.var() != 0 else 1.0
            else:
                gamma_val = float(self.gamma)
            sq_dists = (
                np.sum(X1**2, axis=1)[:, np.newaxis]
                + np.sum(X2**2, axis=1)
                - 2 * np.dot(X1, X2.T)
            )
            return np.exp(-gamma_val * sq_dists)

        else:
            raise ValueError(f"Unsupported kernel: {self.kernel}")

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("SVM currently supports binary classification only.")

        y_binary = np.where(y == self.classes_[1], 1.0, -1.0)

        n_samples, n_features = X.shape
        self.X_train = X
        self.y_train = y_binary

        self.alpha = np.zeros(n_samples)

        K = self._kernel_matrix(X, X)

        for _ in range(self.n_iters):
            for i in range(n_samples):
                gradient = 1.0 - y_binary[i] * np.sum(
                    self.alpha * y_binary * K[:, i]
                )
                self.alpha[i] += self.lr * gradient
                self.alpha[i] = np.clip(self.alpha[i], 0, self.C)

        sv_idx = (self.alpha > 1e-5) & (self.alpha < self.C)
        if np.any(sv_idx):
            self.b = np.mean(
                y_binary[sv_idx]
                - np.dot(K[sv_idx], self.alpha * y_binary)
            )
        else:
            self.b = 0.0

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)

        K_test = self._kernel_matrix(X, self.X_train)

        raw_predictions = np.dot(K_test, self.alpha * self.y_train) + self.b
        binary_preds = np.where(raw_predictions >= 0, 1, 0)
        return self.classes_[binary_preds]