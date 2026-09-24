import numpy as np

class MinMaxScaler:
    def __init__(self, feature_range=(0,1)):
        self.feature_range = feature_range
        self.data_min_ = []
        self.data_max_ = []
        
    def fit(self, X):
        num_features = len(X[0])
        self.data_min_ = [min(row[j] for row in X) for j in range(num_features)]
        self.data_max_ = [max(row[j] for row in X) for j in range(num_features)]
        return self
    
    def transform(self, X):
        a, b = self.feature_range
        X_scaled = []
        
        for row in X:
            scaled_row = []
            for j, val in enumerate(row):
                min_val = self.data_min_[j]
                max_val = self.data_max_[j]
                
                if max_val == min_val:
                    scaled_val = 0
                else:
                    scaled_val = ((val - min_val) / (max_val - min_val)) * (b - a) + a
                    
                scaled_row.append(round(scaled_val, 4))
            X_scaled.append(scaled_row)
        return X_scaled
    
    def fit_transform(self, X):
        return self.fit(X).transform(X)
        
        