import numpy as np

def mean_squared_error(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    
    return np.mean((y_true - y_pred) ** 2)

def mean_absolute_error(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    
    return np.mean(np.abs(y_true - y_pred))

def r2_score(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    
    ss_res = np.sum((y_true - y_pred) ** 2)
    
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    
    return 1.0 - (ss_res / ss_tot)

def confusion_matrix(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    return np.array([[tn, fp], [fn,tp]])

def accuracy_score(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean(y_true == y_pred)

def precision_score(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    
    if tp + fp == 0:
        return 0.0
    return tp / (tp + fp)

def recall_score(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    if tp + fn == 0:
        return 0.0
    return tp / (tp + fn)

def f1_score(y_true, y_pred):
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    
    if p + r == 0:
        return 0.0
    return 2 * (p*r)/(p+r)

def roc_auc_score(y_true, y_score):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    
    desc_score_indices = np.argsort(y_score)[::-1]
    y_true = y_true[desc_score_indices]
    y_score = y_score[desc_score_indices]

    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    tps = np.cumsum(y_true)[threshold_idxs]
    fps = (1 - y_true).cumsum()[threshold_idxs]

    tps = np.r_[0, tps]
    fps = np.r_[0, fps]

    tpr = tps / tps[-1]
    fpr = fps / fps[-1]

    return np.trapz(tpr, fpr)

def log_loss(y_true, y_pred, eps=1e-15):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    y_pred = np.clip(y_pred, eps, 1 - eps)
    
    if y_pred.ndim == 1:
        y_pred = np.c_[1 - y_pred, y_pred]
        
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

def silhouette_score(X, labels):
    X = np.asarray(X)
    labels = np.asarray(labels)
    unique_labels = np.unique(labels)
    n_samples = len(X)

    if len(unique_labels) < 2 or len(unique_labels) >= n_samples:
        raise ValueError("Number of labels is invalid.")

    distances = np.sqrt(((X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2).sum(axis=2))

    a = np.zeros(n_samples)
    b = np.full(n_samples, np.inf)

    for i, label in enumerate(unique_labels):
        cluster_mask = (labels == label)
        if np.sum(cluster_mask) > 1:
            a[cluster_mask] = np.sum(distances[cluster_mask][:, cluster_mask], axis=1) / (np.sum(cluster_mask) - 1)

    for i, label in enumerate(unique_labels):
        cluster_mask = (labels == label)
        for other_label in unique_labels:
            if other_label == label:
                continue
            other_mask = (labels == other_label)
            mean_dist = np.mean(distances[cluster_mask][:, other_mask], axis=1)
            b[cluster_mask] = np.minimum(b[cluster_mask], mean_dist)

    silhouette_values = (b - a) / np.maximum(a, b)
    return np.mean(silhouette_values)