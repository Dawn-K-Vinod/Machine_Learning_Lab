"""Experiment 1 — Linear Regression: Gradient Descent and Normal Equation."""

# --- Part 1 ---
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# --- Part 2 ---
housing = fetch_california_housing()
data = pd.DataFrame(housing.data, columns=housing.feature_names)
data["Price"] = housing.target

X = data[["AveRooms"]].values
y = data["Price"].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Part 3 ---
w = 0
b = 0
learning_rate = 0.01
epochs = 1000
n = len(y_train)
x = X_train_scaled.flatten()

for i in range(epochs):
    y_pred = w * x + b
    dw = (1/n) * np.sum((y_pred - y_train) * x)
    db = (1/n) * np.sum(y_pred - y_train)
    w = w - learning_rate * dw
    b = b - learning_rate * db

y_pred_gd = w * X_test_scaled.flatten() + b
print("Gradient Descent")
print("Weight:", w, "Bias:", b)
print("MSE:", mean_squared_error(y_test, y_pred_gd))
print("R2:", r2_score(y_test, y_pred_gd))

# --- Part 4 ---
X_train_ne = np.c_[np.ones((len(X_train), 1)), X_train]
X_test_ne = np.c_[np.ones((len(X_test), 1)), X_test]
theta = np.linalg.inv(X_train_ne.T @ X_train_ne) @ X_train_ne.T @ y_train
y_pred_ne = X_test_ne @ theta

print("\nNormal Equation")
print("Intercept:", theta[0], "Slope:", theta[1])
print("MSE:", mean_squared_error(y_test, y_pred_ne))
print("R2:", r2_score(y_test, y_pred_ne))
