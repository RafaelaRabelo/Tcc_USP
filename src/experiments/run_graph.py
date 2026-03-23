import pandas as pd
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.config import DATASET_CSV_PATH, EMBEDDING_PATH
from src.features.graph import extract_graph_features
from src.evaluation.metrics import evaluate_classifier
from src.features.embeddings import load_embeddings


def main():
    df = pd.read_csv(DATASET_CSV_PATH)

    # ============================================
    # 🔹 Load embeddings
    # ============================================
    print("📥 Carregando embeddings...")
    model = load_embeddings(str(EMBEDDING_PATH))

    # ============================================
    # 🔹 Graph features (CNE)
    # ============================================
    print("🔗 Extraindo features de grafo com CNE...")
    graph_df = extract_graph_features(
        df,
        embedding_model=model
    )

    # ============================================
    # 🔹 Features
    # ============================================
    X = graph_df.drop(columns=["id", "label"])
    y = graph_df["label"]

    # ============================================
    # 🔥 Pipeline CORRETO
    # ============================================
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf"))
    ])

    # ============================================
    # 🔹 Cross-validation
    # ============================================
    print("🧪 Rodando cross-validation...")
    y_pred = cross_val_predict(clf, X, y, cv=5)

    # ============================================
    # 🔹 Avaliação
    # ============================================
    evaluate_classifier(y, y_pred)


if __name__ == "__main__":
    main()