"""Experiment 5 — KNN Image Classification on Fashion-MNIST."""

# --- Part 1 ---
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# --- Part 2 ---
fashion_mnist = fetch_openml("Fashion-MNIST", version=1, as_frame=False)
X, y = fashion_mnist.data, fashion_mnist.target.astype(int)

X_subset, _, y_subset, _ = train_test_split(
    X, y, train_size=12000, stratify=y, random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(
    X_subset, y_subset, test_size=2000, stratify=y_subset, random_state=42
)
X_train = X_train / 255.0
X_test = X_test / 255.0

# --- Part 3 ---
results = {}
for k in [1, 3, 5, 7, 9, 15]:
    model = KNeighborsClassifier(n_neighbors=k, metric="euclidean", n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results[k] = (accuracy_score(y_test, y_pred), y_pred)
    print(f"K={k}: accuracy={results[k][0]:.4f}")

best_k = max(results, key=lambda k: results[k][0])
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
print(f"\nBest K: {best_k} (accuracy={results[best_k][0]:.4f})")
print(classification_report(y_test, results[best_k][1], target_names=class_names))
