"""
linguistic.py

Extrai features linguísticas simples (MVP), como no seu notebook:
- num_words
- num_unique_words
- lexical_diversity
- hesitations (uh, um)
- avg_word_length
"""

import numpy as np
import pandas as pd


HESITATION_TOKENS = {"uh", "um"}


def _tokenize(text: str) -> list[str]:
    return [t for t in text.split() if t.strip()]


def extract_linguistic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recebe df com colunas [id, text, label] e devolve df com features numéricas.
    """
    rows = []

    for _, row in df.iterrows():
        text = row["text"]
        tokens = _tokenize(text)

        num_words = len(tokens)
        unique_words = set(tokens)
        num_unique = len(unique_words)

        lexical_div = (num_unique / num_words) if num_words > 0 else 0.0

        hesitations = sum(1 for t in tokens if t in HESITATION_TOKENS)

        avg_word_len = float(np.mean([len(t) for t in tokens])) if num_words > 0 else 0.0

        rows.append(
            {
                "id": row["id"],
                "label": row["label"],
                "num_words": num_words,
                "num_unique_words": num_unique,
                "lexical_diversity": lexical_div,
                "hesitations": hesitations,
                "avg_word_length": avg_word_len,
            }
        )

    return pd.DataFrame(rows)