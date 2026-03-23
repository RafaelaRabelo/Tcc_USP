"""
run_fusion.py

🔥 Modelo final:
BoW + Graph + CNE

- Sem leakage
- Cross-validation correto
- Pipeline completo
"""

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_predict
from sklearn.base import BaseEstimator, TransformerMixin

from src.config import DATASET_CSV_PATH, EMBEDDING_PATH
from src.features.bow import make_bow_vectorizer
from src.features.graph import extract_graph_features
from src.features.embeddings import load_embeddings
from src.evaluation.metrics import evaluate_classifier


# ============================================
# 🔥 CUSTOM TRANSFORMER (Graph + CNE)
# ============================================
class GraphFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = pd.DataFrame({
            "id": range(len(X)),
            "text": X,
            "label": 0  # dummy
        })

        graph_df = extract_graph_features(
            df,
            embedding_model=self.embedding_model
        )

        return graph_df.drop(columns=["id", "label"]).to_numpy()


# ============================================
# 🔥 MAIN
# ============================================
def main():
    df = pd.read_csv(DATASET_CSV_PATH)

    X = df["text"].astype(str)
    y = df["label"]

    # ============================================
    # 🔹 Load embeddings
    # ============================================
    print("📥 Carregando embeddings...")
    embedding_model = load_embeddings(str(EMBEDDING_PATH))

    # ============================================
    # 🔥 PIPELINE COMPLETO (SEM LEAKAGE)
    # ============================================
    pipeline = Pipeline([
        ("features", FeatureUnionWrapper(embedding_model)),
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf"))
    ])

    # ============================================
    # 🔹 Cross-validation
    # ============================================
    print("🧪 Rodando cross-validation...")
    y_pred = cross_val_predict(pipeline, X, y, cv=5)

    evaluate_classifier(y, y_pred)


# ============================================
# 🔥 FEATURE UNION (BoW + Graph)
# ============================================
class FeatureUnionWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.bow = make_bow_vectorizer()
        self.graph = GraphFeatureExtractor(embedding_model)

    def fit(self, X, y=None):
        self.bow.fit(X)
        return self

    def transform(self, X):
        X_bow = self.bow.transform(X).toarray()
        X_graph = self.graph.transform(X)

        return np.hstack([X_bow, X_graph])


# ============================================
if __name__ == "__main__":
    main()