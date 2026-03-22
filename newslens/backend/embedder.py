"""
embedder.py
-----------
Shared embedding utility for NewsLens.

Wraps sentence-transformers so both the indexer and the FastAPI
backend use exactly the same model and preprocessing logic.
A single instance is loaded once and reused (model loading takes ~2s).
"""

import os
from functools import lru_cache
from sentence_transformers import SentenceTransformer

# The model produces 384-dimensional vectors and is fast enough
# to run on CPU. Change here if you want a larger/better model —
# just remember to update ENDEE_DIMENSION in index creation too.
DEFAULT_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_DIM = 384  # output dimension of all-MiniLM-L6-v2


@lru_cache(maxsize=1)
def get_model(model_name: str = DEFAULT_MODEL) -> SentenceTransformer:
    """
    Load and cache the embedding model.
    lru_cache ensures we only load it once per process.
    """
    print(f"[embedder] Loading model: {model_name}")
    model = SentenceTransformer(model_name)
    print(f"[embedder] Model ready. Output dim: {model.get_sentence_embedding_dimension()}")
    return model


def embed_text(text: str) -> list[float]:
    """
    Embed a single string. Returns a Python list of floats.
    Used by the FastAPI /search endpoint for query embedding.
    """
    model = get_model()
    vector = model.encode(text, normalize_embeddings=True)
    return vector.tolist()


def embed_batch(texts: list[str], batch_size: int = 64) -> list[list[float]]:
    """
    Embed a list of strings in batches. Returns a list of float lists.
    Used by the indexer when processing articles.

    normalize_embeddings=True converts to unit vectors, which is required
    for cosine similarity to work correctly with Endee.
    """
    model = get_model()
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return [v.tolist() for v in vectors]


def build_article_text(title: str, body: str) -> str:
    """
    Combine title and body into a single string for embedding.
    Giving the title twice gives it slightly more weight in the
    final vector without needing any special weighting logic.
    """
    title = title.strip()
    body = body.strip()
    # Title is repeated to upweight it relative to body
    return f"{title}. {title}. {body[:512]}"
