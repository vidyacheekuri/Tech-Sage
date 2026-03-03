"""
Singleton Model Registry.

All HuggingFace models are loaded exactly ONCE here at application startup.
Every ML module receives model references from this registry — no module ever
calls SentenceTransformer() or pipeline() on its own.

This is critical for production:
- Models are multi-GB; reloading per request would be catastrophic for latency.
- GPU/CPU memory is allocated once and reused.
- Startup is the only slow phase; inference is fast after that.
"""

import logging
import time

from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

from config.settings import get_settings

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Holds all loaded ML models. Instantiated once at app startup."""

    def __init__(self):
        self.embedding_model: SentenceTransformer | None = None
        self.classifier_pipeline = None
        self.summarizer_model = None
        self.summarizer_tokenizer = None
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load_all(self) -> None:
        """Load all models. Call this once during application startup."""
        if self._loaded:
            logger.warning("Models already loaded — skipping redundant load.")
            return

        settings = get_settings()
        total_start = time.time()

        # 1. Sentence embedding model
        logger.info("Loading embedding model: %s", settings.embedding_model)
        start = time.time()
        self.embedding_model = SentenceTransformer(settings.embedding_model)
        logger.info("Embedding model loaded in %.1fs", time.time() - start)

        # 2. Zero-shot classification pipeline
        logger.info("Loading classifier model: %s", settings.classifier_model)
        start = time.time()
        self.classifier_pipeline = pipeline(
            "zero-shot-classification",
            model=settings.classifier_model,
        )
        logger.info("Classifier model loaded in %.1fs", time.time() - start)

        # 3. Summarization model + tokenizer (loaded directly — the "summarization"
        #    pipeline task was removed in transformers >=4.50; loading the model and
        #    tokenizer explicitly is the stable, forward-compatible approach).
        logger.info("Loading summarizer model: %s", settings.summarizer_model)
        start = time.time()
        self.summarizer_tokenizer = AutoTokenizer.from_pretrained(settings.summarizer_model)
        self.summarizer_model = AutoModelForSeq2SeqLM.from_pretrained(settings.summarizer_model)
        logger.info("Summarizer model loaded in %.1fs", time.time() - start)

        self._loaded = True
        logger.info(
            "All models loaded successfully in %.1fs total.",
            time.time() - total_start,
        )


# Module-level singleton — import this from anywhere
model_registry = ModelRegistry()
