"""
metrics.py

Centraliza avaliação para não duplicar código entre experimentos.
"""

from sklearn.metrics import classification_report, accuracy_score


def evaluate_classifier(y_true, y_pred) -> dict:
    """
    Retorna métricas em dict + imprime report formatado.
    """
    report = classification_report(y_true, y_pred, output_dict=False)
    acc = accuracy_score(y_true, y_pred)

    print("\n📊 Classification Report:\n")
    print(report)
    print(f"\n🎯 Accuracy: {acc:.4f}\n")

    return {"accuracy": acc}