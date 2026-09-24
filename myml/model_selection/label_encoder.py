class LabelEncoder:
    def __init__(self):
        self.classes_ = []
        
    def fit(self, y):
        self.classes_ = sorted(list(set(y)))
        return self

    def transform(self, y):
        class_to_index = {cls: i for i, cls in enumerate(self.classes_)}
        return [class_to_index[val] for val in y]
    
    def fit_transform(self, y):
        return self.fit(y).transform(y)
    
    def inverse_transform(self, y):
        return [self.classes_[idx] for idx in y]