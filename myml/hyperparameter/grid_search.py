import itertools
import numpy as np

class GridSearchCV:
    def __init__(self, estimator, param_grid, cv=5, scoring=None):
        self.estimator = estimator
        self.param_grid = param_grid
        self.cv = cv
        self.scoring = scoring

    def _generate_param_combinations(self):
        keys = self.param_grid.keys()
        values = self.param_grid.values()
        for instance in itertools.product(*values):
            yield dict(zip(keys, instance))

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        self.cv_results_ = []
        self.best_score_ = -np.inf
        self.best_params_ = None

        param_combinations = list(self._generate_param_combinations())

        for params in param_combinations:
            estimator = self.estimator.__class__(**params)
            
            if isinstance(self.cv, int):
                n_samples = len(X)
                indices = np.arange(n_samples)
                fold_sizes = np.full(self.cv, n_samples // self.cv, dtype=int)
                fold_sizes[: n_samples % self.cv] += 1
                
                splits = []
                current = 0
                for fold_size in fold_sizes:
                    start, stop = current, current + fold_size
                    val_idx = indices[start:stop]
                    train_idx = np.concatenate([indices[:start], indices[stop:]])
                    splits.append((train_idx, val_idx))
                    current = stop
            else:
                splits = self.cv.split(X)

            scores = []
            for train_idx, val_idx in splits:
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                model = estimator.__class__(**estimator.get_params())
                model.fit(X_train, y_train)

                if self.scoring is None:
                    score = model.score(X_val, y_val)
                else:
                    y_pred = model.predict(X_val)
                    score = self.scoring(y_val, y_pred)

                scores.append(score)

            mean_score = np.mean(scores)
            self.cv_results_.append({
                "params": params,
                "mean_test_score": mean_score,
                "std_test_score": np.std(scores)
            })

            if mean_score > self.best_score_:
                self.best_score_ = mean_score
                self.best_params_ = params

        self.best_estimator_ = self.estimator.__class__(**self.best_params_)
        self.best_estimator_.fit(X, y)
        return self

    def predict(self, X):
        return self.best_estimator_.predict(X)

    def predict_proba(self, X):
        return self.best_estimator_.predict_proba(X)

    def score(self, X, y):
        return self.best_estimator_.score(X, y)