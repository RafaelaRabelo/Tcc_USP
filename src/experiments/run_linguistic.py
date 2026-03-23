"""
run_linguistic.py

Reprodução do baseline com métricas linguísticas.
"""

import pandas as pd

from src.config import DATASET_CSV_PATH, FEATURES_LINGUISTIC_PATH, PROCESSED_DIR
from src.features.linguistic import extract_linguistic_features
from src.models.train_eval import fit_predict_evaluate


def main():
    df = pd.read_csv(DATASET_CSV_PATH)

    features_df = extract_linguistic_features(df)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(FEATURES_LINGUISTIC_PATH, index=False)

    X = features_df.drop(columns=["id", "label"])
    y = features_df["label"]

    fit_predict_evaluate(X, y)


if __name__ == "__main__":
    main()