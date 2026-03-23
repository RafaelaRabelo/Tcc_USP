"""
run_bow.py

Reprodução fiel do Bag of Words (paper)
- CountVectorizer
- SVM (RBF)
- 5-fold Cross Validation
"""

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_predict

from src.config import DATASET_CSV_PATH
from src.features.bow import make_bow_vectorizer
from src.evaluation.metrics import evaluate_classifier


def main():
    df = pd.read_csv(DATASET_CSV_PATH)

    X = df["text"].astype(str)
    y = df["label"]

    # 🔥 Pipeline correto (evita leakage)
    pipeline = Pipeline(
        steps=[
            ("vectorizer", make_bow_vectorizer()),
            ("clf", SVC(kernel="rbf")),
        ]
    )

    # 🔥 Cross-validation (igual paper)
    y_pred = cross_val_predict(pipeline, X, y, cv=5)

    evaluate_classifier(y, y_pred)


if __name__ == "__main__":
    main()