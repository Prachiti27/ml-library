import pandas as pd

from myml.preprocessing import StandardScaler
from myml.pca import PCA
from myml.svm import SVM
from myml.data_splitting import train_test_split
from myml.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

df = pd.read_csv("data/breast_cancer.csv")

X = df[
    [
        "radius",
        "texture",
        "perimeter",
        "area",
        "smoothness",
        "compactness"
    ]
]

y = df["diagnosis"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

pca = PCA(n_components=2)

X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)


print("Original number of features:", X_train.shape[1])
print("Reduced number of features:", X_train_pca.shape[1])

print("\nPCA transformed training data:")
print(X_train_pca[:5])

model = SVM(
    C=1.0,
    kernel="rbf"
)

model.fit(X_train_pca, y_train)

y_pred = model.predict(X_test_pca)

print("\n--- SVM Results ---")

print("Accuracy:",
      accuracy_score(y_test, y_pred))

print("Precision:",
      precision_score(y_test, y_pred))

print("Recall:",
      recall_score(y_test, y_pred))

print("F1 Score:",
      f1_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

new_sample = [[
    14.5,    # radius
    18.2,    # texture
    95.0,    # perimeter
    650.0,   # area
    0.095,   # smoothness
    0.10     # compactness
]]

new_sample_scaled = scaler.transform(new_sample)

new_sample_pca = pca.transform(new_sample_scaled)

prediction = model.predict(new_sample_pca)

print("\nNew sample prediction:", prediction[0])