"""
Embedding generation using sentence-transformers/all-MiniLM-L6-v2.

This module does NOT load any model. It receives the pre-loaded model
from the ModelRegistry singleton.
"""

import logging

import numpy as np

from config.model_registry import model_registry

logger = logging.getLogger(__name__)


def generate_embedding(text: str) -> np.ndarray:
    """
    Generate a 384-dimensional embedding for the given text.

    Args:
        text: Cleaned article text (title + content).

    Returns:
        numpy array of shape (384,).

    Raises:
        RuntimeError: If models have not been loaded via ModelRegistry.
    """
    if not model_registry.is_loaded:
        raise RuntimeError("Models not loaded. Call model_registry.load_all() at startup.")

    if not text or not text.strip():
        logger.warning("Empty text provided for embedding — returning zero vector.")
        return np.zeros(384, dtype=np.float32)

    embedding = model_registry.embedding_model.encode(
        text,
        convert_to_numpy=True,
        normalize_embeddings=True,  # L2 normalization for cosine similarity
        show_progress_bar=False,
    )
    return embedding.astype(np.float32)


def generate_embeddings_batch(texts: list[str]) -> np.ndarray:
    """
    Generate embeddings for a batch of texts.

    Args:
        texts: List of cleaned article texts.

    Returns:
        numpy array of shape (N, 384).
    """
    if not model_registry.is_loaded:
        raise RuntimeError("Models not loaded. Call model_registry.load_all() at startup.")

    if not texts:
        return np.empty((0, 384), dtype=np.float32)

    embeddings = model_registry.embedding_model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=32,
    )
    return embeddings.astype(np.float32)
