"""
make_dataset.py

Constrói o dataset final a partir dos arquivos brutos.

Alinhado com o paper:
- NÃO faz lemmatization
- NÃO faz limpeza agressiva
- Mantém estrutura do texto (importante para grafos)
- Remove apenas textos inválidos

Etapas:
    1) Percorrer arquivos control (label = 0)
    2) Percorrer arquivos dementia (label = 1)
    3) Extrair e limpar texto (leve)
    4) Filtrar textos inválidos
    5) Retornar DataFrame estruturado
"""

import os
import pandas as pd
from tqdm import tqdm

from src.config import (
    CONTROL_PATH,
    DEMENTIA_PATH,
    DATASET_CSV_PATH,
    PROCESSED_DIR
)
from src.preprocessing import extract_par_text, clean_text


# ============================================
# 🔹 Helper: validação de texto
# ============================================
def is_valid_text(text: str, min_tokens: int = 3) -> bool:
    """
    Garante que o texto tem conteúdo suficiente.
    Evita grafos degenerados (problema do paper).
    """
    if not text:
        return False

    tokens = text.split()

    if len(tokens) < min_tokens:
        return False

    return True


# ============================================
# 🔹 Build dataset
# ============================================
def build_dataset() -> pd.DataFrame:
    """
    Constrói o dataset completo (control vs dementia).

    Retorna:
        pd.DataFrame com colunas:
            - id
            - text
            - label
    """
    data = []

    # =========================
    # CONTROL (label = 0)
    # =========================
    for file in tqdm(os.listdir(CONTROL_PATH), desc="Processing CONTROL"):
        file_path = os.path.join(CONTROL_PATH, file)

        try:
            text = extract_par_text(file_path)
            text = clean_text(text)

            if not is_valid_text(text):
                continue

            data.append({
                "id": file,
                "text": text,
                "label": 0
            })

        except Exception as e:
            print(f"Erro em CONTROL {file}: {e}")

    # =========================
    # DEMENTIA (label = 1)
    # =========================
    for file in tqdm(os.listdir(DEMENTIA_PATH), desc="Processing DEMENTIA"):
        file_path = os.path.join(DEMENTIA_PATH, file)

        try:
            text = extract_par_text(file_path)
            text = clean_text(text)

            if not is_valid_text(text):
                continue

            data.append({
                "id": file,
                "text": text,
                "label": 1
            })

        except Exception as e:
            print(f"Erro em DEMENTIA {file}: {e}")

    df = pd.DataFrame(data)

    print("\n📊 Dataset final:")
    print(df["label"].value_counts())
    print(f"Total: {len(df)} amostras")

    return df


# ============================================
# 🔹 Save dataset
# ============================================
def save_dataset(df: pd.DataFrame) -> None:
    """
    Salva o dataset em data/processed/dataset.csv
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df.to_csv(DATASET_CSV_PATH, index=False)

    print(f"\n💾 Dataset salvo em: {DATASET_CSV_PATH}")