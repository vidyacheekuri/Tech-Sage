"""
Zero-shot category classification using facebook/bart-large-mnli.

This module does NOT load any model. It uses the pre-loaded pipeline
from the ModelRegistry singleton.

Categories: AI, Startups, Cybersecurity, Cloud, Big Tech, Gadgets, Open Source, Policy
"""

import logging

from config.model_registry import model_registry
from config.settings import get_settings

logger = logging.getLogger(__name__)


def classify_article(text: str) -> tuple[str, float]:
    """
    Classify an article into one of the predefined tech categories.

    Args:
        text: Article text to classify (title + cleaned content).
              Truncated to first 512 chars for efficiency — BART's NLI
              head works on premise/hypothesis pairs and doesn't need
              the full document.

    Returns:
        Tuple of (predicted_category, confidence_score).
    """
    if not model_registry.is_loaded:
        raise RuntimeError("Models not loaded. Call model_registry.load_all() at startup.")

    settings = get_settings()

    # Truncate input — zero-shot classification doesn't improve with very long text
    input_text = text[:512] if len(text) > 512 else text

    result = model_registry.classifier_pipeline(
        input_text,
        candidate_labels=settings.categories,
        multi_label=False,
    )

    top_category = result["labels"][0]
    top_score = float(result["scores"][0])

    logger.debug(
        "Classification: '%s' -> %s (confidence=%.3f)",
        input_text[:80],
        top_category,
        top_score,
    )

    return top_category, top_score


def classify_articles_batch(texts: list[str]) -> list[tuple[str, float]]:
    """
    Classify a batch of articles.

    Args:
        texts: List of article texts.

    Returns:
        List of (category, confidence) tuples.
    """
    if not model_registry.is_loaded:
        raise RuntimeError("Models not loaded. Call model_registry.load_all() at startup.")

    results = []
    for text in texts:
        results.append(classify_article(text))
    return results
