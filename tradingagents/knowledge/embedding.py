"""Local sentence-transformers embeddings for the semantic half of the search.

Loading the model costs seconds and ~100MB of RSS, so it happens lazily on
first use and is cached per (model, device). Every failure path here degrades to
``None`` rather than raising: an ingestion cycle that cannot embed still stores
full-text-searchable documents, and a search that cannot embed still runs its
keyword arm. Semantic recall is an enhancement, never a dependency.
"""

from __future__ import annotations

import logging
import threading

logger = logging.getLogger(__name__)

_MODELS: dict[str, object] = {}
_MODEL_LOCK = threading.Lock()
# Models that already failed to load, so a search inside a tight loop doesn't
# retry a multi-second import on every call.
_FAILED: set[str] = set()


class Embedder:
    """Wraps a sentence-transformers model with a fixed output width."""

    def __init__(self, model, name: str, dim: int):
        self._model = model
        self.name = name
        self.dim = dim

    def encode(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch. Returns [] on failure so callers can skip vectors."""
        if not texts:
            return []
        try:
            vectors = self._model.encode(
                texts,
                batch_size=32,
                show_progress_bar=False,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
        except Exception as exc:
            logger.warning("embedding failed for %d texts: %s", len(texts), exc)
            return []
        return [[float(x) for x in v] for v in vectors]

    def encode_one(self, text: str) -> list[float] | None:
        vectors = self.encode([text])
        return vectors[0] if vectors else None


def get_embedder(config: dict | None = None) -> Embedder | None:
    """Load the configured embedding model, or return None if unavailable.

    Returns None when embeddings are disabled, when sentence-transformers is not
    installed, or when the model fails to load — all of which are ordinary
    full-text-only operation, not errors.
    """
    if config is None:
        from tradingagents.dataflows.config import get_config

        config = get_config()

    if not config.get("knowledge_embeddings_enabled", True):
        return None

    name = config.get("knowledge_embed_model") or "sentence-transformers/all-MiniLM-L6-v2"
    expected_dim = int(config.get("knowledge_embed_dim", 384))

    with _MODEL_LOCK:
        if name in _FAILED:
            return None
        cached = _MODELS.get(name)
        if cached is not None:
            return cached

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            logger.info(
                "sentence-transformers is not installed; the corpus will run "
                'full-text only. Install with: pip install ".[embeddings]"'
            )
            _FAILED.add(name)
            return None

        try:
            model = SentenceTransformer(name)
            dim = int(model.get_sentence_embedding_dimension())
        except Exception as exc:
            logger.warning("could not load embedding model %r: %s", name, exc)
            _FAILED.add(name)
            return None

        if dim != expected_dim:
            # A mismatch would be rejected by the vector column on insert, with
            # a far less actionable message than this one.
            logger.error(
                "embedding model %r produces %d dimensions but knowledge_embed_dim "
                "is %d; set TRADINGAGENTS_KNOWLEDGE_EMBED_DIM=%d and re-index. "
                "Running full-text only for now.",
                name, dim, expected_dim, dim,
            )
            _FAILED.add(name)
            return None

        embedder = Embedder(model, name, dim)
        _MODELS[name] = embedder
        return embedder


def reset_cache() -> None:
    """Drop cached models and failure memos. For tests and config reloads."""
    with _MODEL_LOCK:
        _MODELS.clear()
        _FAILED.clear()


def to_pgvector(vector: list[float]) -> str:
    """Render a vector in pgvector's text input format."""
    return "[" + ",".join(f"{x:.7g}" for x in vector) + "]"
