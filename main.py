import pandas as pd

from myml.naive_bayes import NaiveBayes
from myml.data_splitting import train_test_split
from myml.preprocessing import StandardScaler
from myml.metrics import (accuracy_score,precision_score,recall_score,f1_score,confusion_matrix)

df = pd.read_csv("data/spam.csv")

print(df.head())

X = df[["word_freq_free","word_freq_money","word_freq_offer","capital_run_length"]]
y = df["spam"]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = NaiveBayes()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

cm = confusion_matrix(y_test, y_pred)

print("\nModel Evaluation")
print("----------------")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

print("\nConfusion Matrix:")
print(cm)

new_email = [[2.5, 1.2, 0.8, 15]]

new_email = scaler.transform(new_email)

prediction = model.predict(new_email)

print("\nNew Email")
print("---------")

if prediction[0] == 1:
    print("Prediction: SPAM")
else:
    print("Prediction: NOT SPAM")