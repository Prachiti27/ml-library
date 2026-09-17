import numpy as np

def train_test_split(*arrays, test_size=None, train_size=None, random_state=None, shuffle=True):
    if not arrays:
        raise ValueError("At least one array required as input.")
    
    arrays = [np.asarray(arr) for arr in arrays]
    n_samples = len(arrays[0])
    
    for i, arr in enumerate(arrays[1:], start=1):
        if len(arr) != n_samples:
            raise ValueError(f"Inconsistent sample counts: Array 0 has {n_samples}, Array {1} has {len(arr)}.")
        
    if test_size is None and train_size is None:
        test_size = 0.25
        
    if test_size is not None:
        if isinstance(test_size, float):
            if not (0.0 < test_size < 1.0):
                raise ValueError("test_size as float must be between 0.0 and 1.0.")
            n_test = int(round(n_samples * test_size))
        elif isinstance(test_size, int):
            n_test = test_size
        else:
            raise TypeError("test_size must be float or int.")
    else:
        n_test = n_samples - (int(round(n_samples * train_size)) if isinstance(train_size, float) else train_size)
        
    if train_size is not None:
        if isinstance(train_size, float):
            n_train = int(round(n_samples * train_size))
        elif isinstance(train_size, int):
            n_train = train_size
        else:
            raise TypeError("train_size must be float or int.")
    else:
        n_train = n_samples - n_test
        
    if n_train + n_test > n_samples:
        raise ValueError(f"Sum of trian ({n_train}) and test({n_test}) samples exceeds dataset size ({n_samples}).")
    
    indices = np.arange(n_samples)
    
    if shuffle:
        rng = np.random.default_rng(random_state)
        rng.shuffle(indices)
        
    train_idx = indices[:n_train]
    test_idx = indices[n_train:n_train + n_test]
    
    split_result = []
    
    for arr in arrays:
        split_result.append(arr[train_idx])
        split_result.append(arr[test_idx])
    
    return split_result