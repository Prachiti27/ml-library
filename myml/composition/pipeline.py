import numpy as np

class Pipeline:
    def __init__(self, steps):
        self.steps = steps

    def _validate_steps(self):
        self.named_steps = dict(self.steps)
        self.transformers = self.steps[:-1]
        self.final_estimator = self.steps[-1][1]

    def fit(self, X, y=None, **fit_params):
        self._validate_steps()
        X_transformed = X
        for name, step in self.transformers:
            if hasattr(step, "fit_transform"):
                X_transformed = step.fit_transform(X_transformed, y)
            else:
                X_transformed = step.fit(X_transformed, y).transform(X_transformed)
        
        if self.final_estimator is not None:
            if y is not None:
                self.final_estimator.fit(X_transformed, y, **fit_params)
            else:
                self.final_estimator.fit(X_transformed, **fit_params)
        return self

    def fit_transform(self, X, y=None, **fit_params):
        self._validate_steps()
        X_transformed = X
        for name, step in self.transformers:
            if hasattr(step, "fit_transform"):
                X_transformed = step.fit_transform(X_transformed, y)
            else:
                X_transformed = step.fit(X_transformed, y).transform(X_transformed)

        if self.final_estimator is None:
            return X_transformed
        if hasattr(self.final_estimator, "fit_transform"):
            return self.final_estimator.fit_transform(X_transformed, y, **fit_params)
        return self.final_estimator.fit(X_transformed, y, **fit_params).transform(X_transformed)

    def predict(self, X):
        X_transformed = X
        for name, step in self.transformers:
            X_transformed = step.transform(X_transformed)
        return self.final_estimator.predict(X_transformed)

    def predict_proba(self, X):
        X_transformed = X
        for name, step in self.transformers:
            X_transformed = step.transform(X_transformed)
        return self.final_estimator.predict_proba(X_transformed)

    def score(self, X, y):
        X_transformed = X
        for name, step in self.transformers:
            X_transformed = step.transform(X_transformed)
        return self.final_estimator.score(X_transformed, y)