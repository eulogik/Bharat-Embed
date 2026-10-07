"""Shared prefix and truncation helpers for Bharat-Embed.

Byte exact vs google/embeddinggemma-2 card.
Text only. No vision or audio paths here.
"""

QUERY_PREFIXES = {
    "SearchQuery": "task: search result | query: {query}",
    "QuestionAnswering": "task: question answering | query: {query}",
    "FactChecking": "task: fact checking | query: {query}",
    "CodeRetrieval": "task: code retrieval | query: {query}",
    "Classification": "task: classification | query: {content}",
    "Clustering": "task: clustering | query: {content}",
    "SentenceSimilarity": "task: sentence similarity | query: {content}",
}

SUPPORTED_MRL_DIMS = [128, 256, 512, 768]
DEFAULT_MRL_TRAIN = [128, 256, 512, 768]
LAB_MRL_32 = 32


def format_query(prompt_name: str, text: str) -> str:
    if prompt_name not in QUERY_PREFIXES:
        raise ValueError(f"unknown prompt_name: {prompt_name}")
    template = QUERY_PREFIXES[prompt_name]
    if "{query}" in template:
        return template.replace("{query}", text)
    return template.replace("{content}", text)


def format_doc(title: str | None, content: str) -> str:
    t = title.strip() if title and title.strip() else "none"
    return f"title: {t} | text: {content}"


def check_same_dim(query_dim: int, doc_dim: int) -> None:
    if query_dim != doc_dim:
        raise ValueError(f"query dim {query_dim} != doc dim {doc_dim}. Keep both same.")


def truncate_and_renorm(vec, dim: int):
    import numpy as np

    if dim not in SUPPORTED_MRL_DIMS and dim != LAB_MRL_32:
        raise ValueError(f"bad dim {dim}. Supported: {SUPPORTED_MRL_DIMS} plus lab 32.")
    short = np.asarray(vec, dtype=np.float64)[:dim]
    norm = float(np.linalg.norm(short))
    if norm == 0.0:
        raise ValueError("zero norm, cannot renorm.")
    return (short / norm).astype(float)


def pick_dtype(train_on_mps: bool = False) -> str:
    # Safe default. bf16 only where proven.
    # Base card warns fp16 gives NaN, so fp16 is never allowed here.
    try:
        import torch

        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return "bfloat16"
    except Exception:
        pass
    return "float32"
