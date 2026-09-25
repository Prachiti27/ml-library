import numpy as np
from .k_fold import KFold

def CrossValScore(estimator, X, y, cv=5, scoring=None):
    if isinstance(cv, int):
        cv = KFold(n_splits=cv)

    scores = []
    for train_idx, val_idx in cv.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = estimator.__class__(**estimator.get_params())
        model.fit(X_train, y_train)

        if scoring is None:
            score = model.score(X_val, y_val)
        else:
            y_pred = model.predict(X_val)
            score = scoring(y_val, y_pred)

        scores.append(score)

    return np.array(scores)