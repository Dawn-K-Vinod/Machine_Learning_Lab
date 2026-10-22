# Machine Learning Lab — Minimal Exam Versions

This folder contains compact, plot-free versions of the six experiments from the supplied lab notebooks. The algorithms, essential preprocessing, train/test split, and main evaluations are retained; decorative plots, repeated displays, progress logs, and redundant diagnostics are removed.

## Files

Each experiment is provided as both a Jupyter notebook (`.ipynb`) and a plain Python script (`.py`). Run notebook cells from top to bottom, or run the corresponding script with Python.

1. **Experiment 1:** Linear Regression using Gradient Descent and the Normal Equation
2. **Experiment 2:** Linear and Polynomial Regression on Auto MPG
3. **Experiment 3:** Logistic Regression comparison — MLE, MAP-L2, MAP-L1
4. **Experiment 4:** Multinomial and Bernoulli Naive Bayes on 20 Newsgroups
5. **Experiment 5:** KNN image classification on Fashion-MNIST
6. **Experiment 6:** Customer segmentation with an entropy-based (ID3-style) Decision Tree

## Requirements

Install the common dependencies:

```bash
python -m pip install numpy pandas scikit-learn openpyxl jupyter
```

Experiments 1–5 download data on first use, so an internet connection may be needed. Experiment 2 reads the Auto MPG CSV from GitHub. Experiment 6 expects **`Online Retail.xlsx`** to be in the current working directory when you run it (or change the path in the code). That Excel file was not present in the provided ZIP, so it is not included here.

## Important notes

- Plots were removed as requested; core algorithms and main performance metrics remain.
- Experiment 1 uses scaled features for gradient descent and raw features for the normal equation, as in the original experiment.
- Experiment 3 uses the `penalty` argument to make the MLE/L2/L1 comparison explicit. Some recent scikit-learn versions show deprecation warnings for this argument, but the code executes in the version tested. If your environment rejects `penalty=None`, use a compatible scikit-learn version or adapt the model options to that installed version.
- Experiment 5 uses a stratified subset of 12,000 images (10,000 train / 2,000 test) to keep the practical exercise smaller than running the full dataset.
- The term “ID3-style” is used because the decision tree is trained with entropy; scikit-learn's `DecisionTreeClassifier` is not a literal implementation of the original ID3 algorithm.
