"""
Abstractive summarization using facebook/bart-large-cnn.

This module does NOT load any model. It uses the pre-loaded model and tokenizer
from the ModelRegistry singleton.

Uses model.generate() directly instead of the HuggingFace pipeline() wrapper,
because the "summarization" task alias was removed in transformers >=4.50.

Target: ~120-150 word summaries (3-4 informative lines).
"""

import logging

from config.model_registry import model_registry

logger = logging.getLogger(__name__)

_MAX_SUMMARY_TOKENS = 200
_MIN_SUMMARY_TOKENS = 80


def summarize_article(text: str) -> str:
    """
    Generate an abstractive summary of approximately 120-150 words.

    Args:
        text: Cleaned article content. Truncated to ~4000 chars to stay
              within BART-CNN's 1024 token input limit while providing
              more context for richer summaries.

    Returns:
        Summary string.
    """
    if not model_registry.is_loaded:
        raise RuntimeError("Models not loaded. Call model_registry.load_all() at startup.")

    if not text or not text.strip():
        logger.warning("Empty text provided for summarization.")
        return ""

    input_text = text[:4000] if len(text) > 4000 else text

    word_count = len(input_text.split())
    if word_count < 40:
        logger.debug("Text too short for summarization (%d words) — returning as-is.", word_count)
        return input_text

    tokenizer = model_registry.summarizer_tokenizer
    model = model_registry.summarizer_model

    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        max_length=1024,
        truncation=True,
    )

    summary_ids = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=_MAX_SUMMARY_TOKENS,
        min_length=_MIN_SUMMARY_TOKENS,
        num_beams=4,
        length_penalty=1.5,
        do_sample=False,
    )

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    logger.debug("Summary generated: %d words from %d word input.", len(summary.split()), word_count)

    return summary


def summarize_articles_batch(texts: list[str]) -> list[str]:
    """
    Summarize a batch of articles.

    Args:
        texts: List of article contents.

    Returns:
        List of summary strings.
    """
    if not model_registry.is_loaded:
        raise RuntimeError("Models not loaded. Call model_registry.load_all() at startup.")

    return [summarize_article(text) for text in texts]
