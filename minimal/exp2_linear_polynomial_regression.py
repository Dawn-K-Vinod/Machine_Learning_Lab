"""Experiment 2 — Linear and Polynomial Regression on Auto MPG."""

# --- Part 1 ---
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

# --- Part 2 ---
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mpg.csv"
df = pd.read_csv(url)
data = df[["displacement", "mpg"]].dropna()
X = data[["displacement"]]
y = data["mpg"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- Part 3 ---
results = []

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred = linear_model.predict(X_test)
results.append(["Linear", 1, mean_squared_error(y_test, y_pred), r2_score(y_test, y_pred)])

for degree in range(2, 6):
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    y_pred = model.predict(X_test_poly)
    results.append(["Polynomial", degree, mean_squared_error(y_test, y_pred), r2_score(y_test, y_pred)])

print(pd.DataFrame(results, columns=["Model", "Degree", "MSE", "R2"]).to_string(index=False))
