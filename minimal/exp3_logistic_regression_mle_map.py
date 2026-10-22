"""Experiment 3 — Logistic Regression: MLE vs MAP (L1 and L2)."""

# --- Part 1 ---
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# --- Part 2 ---
data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42, stratify=data.target
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

models = {
    "MLE": LogisticRegression(penalty=None, max_iter=5000, random_state=42),
    "MAP (L2)": LogisticRegression(penalty="l2", C=1.0, max_iter=5000, random_state=42),
    "MAP (L1)": LogisticRegression(penalty="l1", solver="liblinear", C=1.0, max_iter=5000, random_state=42),
}

# --- Part 3 ---
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"\n{name}")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1:", f1_score(y_test, y_pred))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    print("Zero coefficients:", np.sum(model.coef_ == 0))
