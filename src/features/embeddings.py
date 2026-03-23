from gensim.models import KeyedVectors


def load_embeddings(path: str):
    print(f"📥 Carregando embeddings: {path}")

    model = KeyedVectors.load_word2vec_format(path, binary=False)

    print("✅ Embeddings carregados!")
    print(f"📊 Vocabulário: {len(model)} palavras")

    return model