import numpy as np

class ColumnTransformer:
    def __init__(self, transformers, remainder="drop"):
        self.transformers = transformers
        self.remainder = remainder

    def fit(self, X, y=None):
        X = np.asarray(X)
        self.transformers_ = []
        fitted_columns = []

        for name, transformer, columns in self.transformers:
            if transformer != "drop":
                fitted_transformer = transformer.fit(X[:, columns], y) if transformer != "passthrough" else "passthrough"
                self.transformers_.append((name, fitted_transformer, columns))
            fitted_columns.extend(columns if isinstance(columns, list) else [columns])

        all_columns = list(range(X.shape[1]))
        self.remainder_cols_ = [col for col in all_columns if col not in fitted_columns]
        return self

    def transform(self, X):
        X = np.asarray(X)
        output_blocks = []

        for name, transformer, columns in self.transformers_:
            X_block = X[:, columns]
            if transformer == "passthrough":
                output_blocks.append(X_block)
            elif transformer != "drop":
                transformed = transformer.transform(X_block)
                if transformed.ndim == 1:
                    transformed = transformed[:, np.newaxis]
                output_blocks.append(transformed)

        if self.remainder == "passthrough" and len(self.remainder_cols_) > 0:
            output_blocks.append(X[:, self.remainder_cols_])

        return np.hstack(output_blocks) if output_blocks else np.empty((len(X), 0))

    def fit_transform(self, X, y=None):
        X = np.asarray(X)
        self.transformers_ = []
        fitted_columns = []
        output_blocks = []

        for name, transformer, columns in self.transformers:
            X_block = X[:, columns]
            if transformer == "passthrough":
                self.transformers_.append((name, "passthrough", columns))
                output_blocks.append(X_block)
            elif transformer != "drop":
                if hasattr(transformer, "fit_transform"):
                    transformed = transformer.fit_transform(X_block, y)
                else:
                    transformed = transformer.fit(X_block, y).transform(X_block)
                
                self.transformers_.append((name, transformer, columns))
                if transformed.ndim == 1:
                    transformed = transformed[:, np.newaxis]
                output_blocks.append(transformed)
            
            fitted_columns.extend(columns if isinstance(columns, list) else [columns])

        all_columns = list(range(X.shape[1]))
        self.remainder_cols_ = [col for col in all_columns if col not in fitted_columns]

        if self.remainder == "passthrough" and len(self.remainder_cols_) > 0:
            output_blocks.append(X[:, self.remainder_cols_])

        return np.hstack(output_blocks) if output_blocks else np.empty((len(X), 0))