import re


# ============================================
# 🔹 EXTRAÇÃO (*PAR:)
# ============================================
def extract_par_text(file_path: str) -> str:
    """
    Extrai apenas fala do participante (*PAR:)
    """
    chunks = []

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("*PAR:"):
                line = line.replace("*PAR:", "").strip()
                chunks.append(line)

    return " ".join(chunks)


# ============================================
# 🔹 LIMPEZA LEVE (ALINHADA AO PAPER)
# ============================================
def clean_text(text: str) -> str:
    """
    Limpeza MINIMALISTA (igual filosofia do paper)

    ✔ remove:
    - tags <...>
    - colchetes CHAT
    - tokens estranhos
    - tabs

    ✔ mantém:
    - estrutura de palavras
    - contrações (don't, he's)
    - palavras com ruído
    """

    # remove tags tipo <...>
    text = re.sub(r"<.*?>", " ", text)

    # remove colchetes CHAT: [//], [: ...]
    text = re.sub(r"\[.*?\]", " ", text)

    # remove tokens estranhos \x15...\x15
    text = re.sub(r"\x15.*?\x15", " ", text)

    # remove TAB
    text = text.replace("\t", " ")

    # 🔥 NÃO remover tudo que não é letra
    # manter apóstrofos (importantíssimo)
    text = re.sub(r"[^a-zA-Z'\s]", " ", text)

    # normaliza espaços
    text = re.sub(r"\s+", " ", text).strip()

    return text.lower()