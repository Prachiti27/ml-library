import numpy as np
from myml.data_splitting import train_test_split

X = np.arange(20).reshape((10,2))
y = np.arange(10)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42,shuffle=True)

print("X_train:\n", X_train)
print("X_test:\n", X_test)
print("y_train:", y_train)
print("y_test:", y_test)