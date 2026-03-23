"""
main.py

Orquestrador do projeto (ALINHADO COM O PAPER):

1) Dataset
2) Experimentos:
   - EXP 1: Linguistic
   - EXP 2: BoW
   - EXP 3: Graph (co-occurrence)
   - EXP 4: CNE (Graph + Embeddings)
"""

import os
import sys

from src.config import DATASET_CSV_PATH
from src.data.make_dataset import build_dataset, save_dataset

# garante que a raiz do projeto está no path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)


def ensure_dataset():
    if os.path.exists(DATASET_CSV_PATH):
        print("✅ dataset.csv já existe. Pulando build.")
        return

    print("📦 Construindo dataset...")
    df = build_dataset()
    save_dataset(df)
    print(f"✅ Dataset salvo em: {DATASET_CSV_PATH}")


def run_experiments():
    # ============================================
    # EXP 1
    # ============================================
    print("\n==============================")
    print("🧪 EXP 1: Linguistic Features")
    print("==============================")
    from src.experiments.run_linguistic import main as exp1
    exp1()

    # ============================================
    # EXP 2
    # ============================================
    print("\n==============================")
    print("🧪 EXP 2: Bag of Words (BoW)")
    print("==============================")
    from src.experiments.run_bow import main as exp2
    exp2()

    # ============================================
    # EXP 3
    # ============================================
    print("\n==============================")
    print("🧪 EXP 3: Graph (Co-occurrence)")
    print("==============================")
    from src.experiments.run_graph_base import main as exp3
    exp3()

    # ============================================
    # EXP 4 (CNE)
    # ============================================
    print("\n==============================")
    print("🧪 EXP 4: CNE (Graph + Embeddings)")
    print("==============================")
    from src.experiments.run_graph import main as exp4
    exp4()


def main():
    print("🚀 Iniciando pipeline...")
    ensure_dataset()
    run_experiments()
    print("\n✅ Pipeline finalizado!")


if __name__ == "__main__":
    main()