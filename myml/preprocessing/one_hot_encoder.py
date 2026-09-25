class OneHotEncoder:
    def __init__(self, handle_unknown='ignore'):
        self.handle_unknown = handle_unknown
        self.categories_ = []
        
    def fit(self, X):
        num_features = len(X[0])
        self.categories_ = []
        for j in range(num_features):
            col = [row[j] for row in X]
            unique_cats = sorted(list(set(col)))
            self.categories_.append(unique_cats)
        return self
    
    def transform(self, X):
        encoded_X = []
        for row in X:
            encoded_row = []
            for j, val in enumerate(row):
                cats = self.categories_[j]
                one_hot = [0] * len(cats)
                if val in cats:
                    one_hot[cats.index(val)] = 1
                elif self.handle_unknown != 'ignore':
                    raise ValueError(f"Unknown category {val} encountered in feature {j}.")
                encoded_row.extend(one_hot)
            encoded_X.append(encoded_row)
        return encoded_X
    
    def fit_transform(self, X):
        return self.fit(X).transform(X)