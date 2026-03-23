"""
config.py

Centraliza caminhos e configurações do projeto.

Você já organizou assim:
- data/raw/control
- data/raw/dementia
- data/processed

Então aqui a gente só "aponta" pra essas pastas.
"""

from pathlib import Path

# Raiz do projeto (pasta onde está o main.py)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

CONTROL_PATH = RAW_DIR / "control"
DEMENTIA_PATH = RAW_DIR / "dementia"

DATASET_CSV_PATH = PROCESSED_DIR / "dataset.csv"
FEATURES_LINGUISTIC_PATH = PROCESSED_DIR / "features_linguistic.csv"
FEATURES_GRAPH_PATH = PROCESSED_DIR / "features_graph.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.2

EMBEDDING_PATH = PROJECT_ROOT / "src" / "models" / "wiki-news-300d-1M.vec"