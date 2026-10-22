"""Experiment 6 — Customer Segmentation using an ID3-style Decision Tree."""

# --- Part 1 ---
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# --- Part 2 ---
df = pd.read_excel("Online Retail.xlsx")
df = df.dropna(subset=["CustomerID"])
df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
df["TotalAmount"] = df["Quantity"] * df["UnitPrice"]

customers = df.groupby("CustomerID").agg(
    TotalSpend=("TotalAmount", "sum"),
    TotalQuantity=("Quantity", "sum"),
    NumberOfTransactions=("InvoiceNo", "nunique"),
    NumberOfProducts=("StockCode", "nunique"),
    AverageOrderValue=("TotalAmount", "mean"),
).reset_index()
customers["Segment"] = pd.qcut(
    customers["TotalSpend"], q=3, labels=["Low Value", "Medium Value", "High Value"]
)

# --- Part 3 ---
features = ["TotalQuantity", "NumberOfTransactions", "NumberOfProducts", "AverageOrderValue"]
X = customers[features]
y = customers["Segment"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = DecisionTreeClassifier(criterion="entropy", max_depth=4, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
