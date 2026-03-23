"""
train_eval.py

Treino e teste padronizados para todos os experimentos.
"""

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from src.config import TEST_SIZE, RANDOM_STATE
from src.evaluation.metrics import evaluate_classifier


def split_xy(X, y):
    return train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )


def train_logistic_regression(X_train, y_train):
    """
    Modelo baseline simples e reproduzível.
    """
    model = LogisticRegression(max_iter=2000)
    model.fit(X_train, y_train)
    return model


def fit_predict_evaluate(X, y):
    """
    Split -> Train -> Predict -> Evaluate
    """
    X_train, X_test, y_train, y_test = split_xy(X, y)
    model = train_logistic_regression(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = evaluate_classifier(y_test, y_pred)
    return model, metrics