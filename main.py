import numpy as np
from myml.data_splitting import train_test_split
from myml.preprocessing import StandardScaler
from myml.linear_regression import LinearRegression
from myml.metrics import mean_squared_error, mean_absolute_error, r2_score

X = np.arange(20).reshape((10,2))
y = np.arange(10)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42,shuffle=True)

print("X_train:\n", X_train)
print("X_test:\n", X_test)
print("y_train:", y_train)
print("y_test:", y_test)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("X_train: \n", X_train)
print("X_test: \n", X_test)

model = LinearRegression(method="gd", lr=0.01, epochs=2000)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("MSE: ", mean_squared_error(y_test, y_pred))
print("MAE: ", mean_absolute_error(y_test, y_pred))
print("R2: ", r2_score(y_test, y_pred))