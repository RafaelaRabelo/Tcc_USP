import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import skew


# ============================================
# 🔹 CONFIG (balanceado)
# ============================================
MAX_NODES = 150            # 🔥 maior contexto
WINDOW_SIZE = 3            # 🔥 essencial
EMBEDDING_THRESHOLD = 0.3  # 🔥 mais conexões


# ============================================
# 🔹 TOKEN CLEANING
# ============================================
def clean_tokens(text):
    return [
        t.lower()
        for t in text.split()
        if len(t) > 2 and t.isalpha()
    ][:MAX_NODES]


# ============================================
# 🔹 BASE GRAPH (WINDOW)
# ============================================
def build_word_graph(tokens):
    G = nx.Graph()

    for i in range(len(tokens)):
        for j in range(i + 1, min(i + WINDOW_SIZE, len(tokens))):
            if tokens[i] != tokens[j]:
                G.add_edge(tokens[i], tokens[j])

    return G


# ============================================
# 🔹 CNE (embedding enrichment)
# ============================================
def enrich_graph_with_embeddings(G, model):
    words = list(G.nodes())

    for i in range(len(words)):
        for j in range(i + 1, len(words)):
            w1, w2 = words[i], words[j]

            if w1 in model and w2 in model:
                vec1 = model[w1]
                vec2 = model[w2]

                denom = np.linalg.norm(vec1) * np.linalg.norm(vec2)
                if denom == 0:
                    continue

                sim = np.dot(vec1, vec2) / denom

                if sim >= EMBEDDING_THRESHOLD:
                    G.add_edge(w1, w2)

    return G


# ============================================
# 🔹 GRAPH METRICS (mais ricas)
# ============================================
def extract_graph_metrics(G):

    if G.number_of_nodes() == 0:
        return {
            "degree": [0],
            "pagerank": [0],
            "clustering": [0],
            "betweenness": [0],
            "density": [0],
        }

    try:
        metrics = {
            "degree": list(dict(G.degree()).values()),
            "pagerank": list(nx.pagerank(G).values()),
            "clustering": list(nx.clustering(G).values()),
            "betweenness": list(nx.betweenness_centrality(G).values()),
            "density": [nx.density(G)],
        }

        # opcional (cuidado com custo)
        if nx.is_connected(G):
            metrics["path_length"] = [nx.average_shortest_path_length(G)]
        else:
            metrics["path_length"] = [0]

        return metrics

    except:
        return {
            "degree": [0],
            "pagerank": [0],
            "clustering": [0],
            "betweenness": [0],
            "density": [0],
            "path_length": [0],
        }


# ============================================
# 🔹 AGGREGATION (robusta)
# ============================================
def aggregate_metrics(metrics):
    features = []

    for values in metrics.values():
        values = np.array(values)

        if len(values) == 0:
            features.extend([0.0, 0.0, 0.0])
            continue

        mean = np.mean(values)
        std = np.std(values)

        if len(values) < 3 or np.all(values == values[0]):
            skewness = 0.0
        else:
            skewness = skew(values)
            if np.isnan(skewness):
                skewness = 0.0

        features.extend([mean, std, skewness])

    return features


# ============================================
# 🔥 MAIN FUNCTION
# ============================================
def extract_graph_features(df, embedding_model=None):
    rows = []

    for _, row in df.iterrows():

        tokens = clean_tokens(row["text"])

        if len(tokens) < 2:
            continue

        G = build_word_graph(tokens)

        if embedding_model is not None:
            G = enrich_graph_with_embeddings(G, embedding_model)

        metrics = extract_graph_metrics(G)
        features = aggregate_metrics(metrics)

        rows.append({
            "id": row["id"],
            "label": row["label"],
            **{f"f_{i}": v for i, v in enumerate(features)}
        })

    df_features = pd.DataFrame(rows)

    df_features = df_features.replace([np.inf, -np.inf], 0)
    df_features = df_features.fillna(0)

    return df_features