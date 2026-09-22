import pandas as pd

from myml.decision_tree import DecisionTreeRegressor
from myml.data_splitting import train_test_split
from myml.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

df = pd.read_csv("data/house_prices.csv")
print(df.head())

X = df[["area","bedrooms","age"]]
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model = DecisionTreeRegressor(max_depth=4,min_samples_split=2)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)

mae = mean_absolute_error(y_test, y_pred)

r2 = r2_score(y_test, y_pred)


print("\nModel Evaluation")
print("----------------")

print(f"MSE : {mse:.4f}")
print(f"MAE : {mae:.4f}")
print(f"R²  : {r2:.4f}")

print("\nActual vs Predicted")
print("-------------------")

for actual, predicted in zip(y_test, y_pred):

    print(
        f"Actual: {actual:.2f} "
        f"| Predicted: {predicted:.2f}"
    )

new_house = [[2300, 3, 2]]

prediction = model.predict(new_house)


print("\nNew House")
print("---------")

print("Area     : 2300 sq ft")
print("Bedrooms : 3")
print("Age      : 2 years")

print(
    f"Predicted Price: ₹{prediction[0]:.2f} lakh"
)