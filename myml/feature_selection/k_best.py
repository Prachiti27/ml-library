import numpy as np

def f_classif(X, y):
    X = np.asarray(X)
    y = np.asarray(y)
    classes = np.unique(y)
    n_samples, n_features = X.shape
    n_classes = len(classes)
    
    grand_mean = np.mean(X, axis=0)
    ss_between = np.zeros(n_features)
    ss_within = np.zeros(n_features)
    
    for c in classes:
        X_c = X[y == c]
        n_c = len(X_c)
        mean_c = np.mean(X_c, axis=0)
        ss_between += n_c * (mean_c - grand_mean) ** 2
        ss_within += np.sum((X_c - mean_c) ** 2, axis=0)
        
    df_between = n_classes - 1
    df_within = n_samples - n_classes
    
    ms_between = ss_between / df_between
    ms_within = ss_within / df_within
    
    f_statistic = ms_between / ms_within
    return f_statistic

class SelectKBest:
    def __init__(self, score_func=f_classif, k=10):
        self.score_func = score_func
        self.k = k

    def fit(self, X, y):
        X = np.asarray(X)
        self.scores_ = self.score_func(X, y)
        return self

    def transform(self, X):
        X = np.asarray(X)
        mask = self.get_support()
        return X[:, mask]

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)

    def get_support(self, indices=False):
        n_features = len(self.scores_)
        k = min(self.k, n_features)
        
        mask = np.zeros(n_features, dtype=bool)
        top_indices = np.argsort(self.scores_)[::-1][:k]
        
        if indices:
            return top_indices
        
        mask[top_indices] = True
        return mask