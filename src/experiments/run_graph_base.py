import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_predict

from src.config import DATASET_CSV_PATH
from src.features.graph import extract_graph_features
from src.evaluation.metrics import evaluate_classifier


def main():
    df = pd.read_csv(DATASET_CSV_PATH)

    print("🔗 Extraindo grafo base (sem embeddings)...")
    graph_df = extract_graph_features(df, embedding_model=None)

    X = graph_df.drop(columns=["id", "label"])
    y = graph_df["label"]

    clf = SVC(kernel="rbf")

    y_pred = cross_val_predict(clf, X, y, cv=5)

    evaluate_classifier(y, y_pred)


if __name__ == "__main__":
    main()