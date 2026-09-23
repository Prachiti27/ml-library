import pandas as pd

from myml.k_means import Kmeans
from myml.preprocessing import StandardScaler

df = pd.read_csv("data/customer_segments.csv")

X = df[["annual_income", "spending_score"]]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = Kmeans(k=3,max_iters=100,random_state=42)

model.fit(X_scaled)

clusters = model.predict(X_scaled)

df["cluster"] = clusters

print(df)

new_customer = [[85, 72]]

new_customer_scaled = scaler.transform(new_customer)

prediction = model.predict(new_customer_scaled)

print("\nNew customer cluster:", prediction[0])