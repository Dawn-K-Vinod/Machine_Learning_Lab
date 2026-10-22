"""Experiment 4 — Naive Bayes Text Classification on 20 Newsgroups."""

# --- Part 1 ---
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# --- Part 2 ---
train_data = fetch_20newsgroups(subset="train", remove=("headers", "footers", "quotes"))
test_data = fetch_20newsgroups(subset="test", remove=("headers", "footers", "quotes"))
X_train_text, y_train = train_data.data, train_data.target
X_test_text, y_test = test_data.data, test_data.target

vectorizer = CountVectorizer(stop_words="english", min_df=2)
X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)

binary_vectorizer = CountVectorizer(stop_words="english", min_df=2, binary=True)
X_train_binary = binary_vectorizer.fit_transform(X_train_text)
X_test_binary = binary_vectorizer.transform(X_test_text)

# --- Part 3 ---
models = [
    ("Multinomial NB", MultinomialNB(), X_train, X_test),
    ("Bernoulli NB", BernoulliNB(), X_train_binary, X_test_binary),
]

for name, model, Xtr, Xte in models:
    model.fit(Xtr, y_train)
    y_pred = model.predict(Xte)
    print(f"\n{name}")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision (macro):", precision_score(y_test, y_pred, average="macro"))
    print("Recall (macro):", recall_score(y_test, y_pred, average="macro"))
    print("F1 (macro):", f1_score(y_test, y_pred, average="macro"))
