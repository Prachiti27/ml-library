import numpy as np

class RandomizedSearchCV:
    def __init__(self, estimator, param_distributions, n_iter=10, cv=5, scoring=None, random_state=None):
        self.estimator = estimator
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.cv = cv
        self.scoring = scoring
        self.random_state = random_state

    def _sample_params(self, rng):
        params = {}
        for k, v in self.param_distributions.items():
            if hasattr(v, "rvs"):
                params[k] = v.rvs(random_state=rng)
            elif isinstance(v, list):
                params[k] = rng.choice(v)
            else:
                params[k] = v
        return params

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        rng = np.random.RandomState(self.random_state)

        self.cv_results_ = []
        self.best_score_ = -np.inf
        self.best_params_ = None

        param_combinations = [self._sample_params(rng) for _ in range(self.n_iter)]

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