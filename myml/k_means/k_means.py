import numpy as np

class Kmeans:
    def __init__(self, k=3, max_iters=100, tol=1e-4, random_state=None):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.random_state = random_state
        self.centroids = None
        
    def _init_centroids(self, X):
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, size=self.k, replace=False)
        return X[indices]
    
    def _compute_distances(self, X):
        return np.linalg.norm(X[:, np.newaxis, :] - self.centroids, axis=2)
    
    def fit(self, X):
        if self.random_state is not None:
            np.random.seed(self.random_state)
            
        X = np.asarray(X)
        
        self.centroids = self._init_centroids(X)
        
        for _ in range(self.max_iters):
            distances = self._compute_distances(X)
            labels = np.argmin(distances, axis=1)
            
            new_centroids = np.zeros_like(self.centroids)
            for cluster_idx in range(self.k):
                cluster_points = X[labels == cluster_idx]
                if len(cluster_points) == 0:
                    new_centroids[cluster_idx] = X[np.random.choice(X.shape[0])]
                else:
                    new_centroids[cluster_idx] = cluster_points.mean(axis=0)
                    
            centroid_shift = np.linalg.norm(new_centroids - self.centroids)
            self.centroids = new_centroids
            
            if centroid_shift < self.tol:
                break
        return self
    
    def predict(self, X):
        X = np.asarray(X)
        distances = self._compute_distances(X)
        return np.argmin(distances, axis=1)
    
    def fit_predict(self, X):
        self.fit(X)
        return self.predict(X)

    @property
    def inertia_(self):
        if self.centroids is None:
            raise ValueError("Model is not fitted yet.")
        distances = self._compute_distances(self._X)
        min_distances = np.min(distances, axis=1)
        return np.sum(min_distances ** 2)