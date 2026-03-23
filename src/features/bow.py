"""
bow.py

Bag of Words / TF-IDF (texto -> vetor)

Este módulo oferece DUAS abordagens:

1) 🔬 Baseline fiel ao paper (CountVectorizer)
2) 🚀 Versão melhorada (TF-IDF)

Use a baseline para reproduzir o paper corretamente.
Use a versão TF-IDF para melhorar performance no TCC.
"""

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


# =====================================================
# 🔬 1. BASELINE FIEL AO PAPER
# =====================================================
def make_bow_vectorizer():
    """
    Bag of Words clássico (mais fiel ao paper).

    Características:
    - unigram only
    - sem remoção agressiva
    - mantém ruído (importante no paper)
    """
    return CountVectorizer(
        ngram_range=(1, 1),   # apenas unigram
        stop_words=None       # NÃO remover agressivamente
    )


# =====================================================
# 🚀 2. TF-IDF (VERSÃO MELHORADA)
# =====================================================
def make_tfidf_vectorizer():
    """
    TF-IDF levemente controlado (melhor para performance).

    Ainda relativamente fiel, mas mais forte que o paper.
    """
    return TfidfVectorizer(
        ngram_range=(1, 1),   # manter unigram para comparabilidade
        min_df=1,
        max_df=1.0,
        stop_words=None
    )


# =====================================================
# 🔥 3. TF-IDF OTIMIZADO (SEU MODELO ANTIGO)
# =====================================================
def make_tfidf_optimized():
    """
    Versão mais forte (NÃO usada para reproduzir paper).

    Use isso como "melhor modelo" no TCC.
    """
    return TfidfVectorizer(
        ngram_range=(1, 2),   # bigrams
        min_df=2,
        max_df=0.95,
        stop_words="english"
    )