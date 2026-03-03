"""
TechSage — Secondary release classification layer.

After the primary topic classification, this module runs a second zero-shot pass
to detect whether an article is about a specific type of AI release.

Uses the same BART-MNLI model (already loaded in ModelRegistry) — no extra model loading.

Labels:
  Model Release, Framework Release, Agent Platform,
  Infrastructure Update, Research Paper, General AI News

If the detected label is a release type (not General AI News),
is_release=True and release_type is set accordingly.
"""

import logging

from config.model_registry import model_registry
from config.settings import get_settings

logger = logging.getLogger(__name__)

# Maps zero-shot labels to database release_type values
_LABEL_TO_RELEASE_TYPE = {
    "Model Release": "model",
    "Framework or Library Release and Update": "framework",
    "Agent Platform": "agent_platform",
    "Infrastructure Update": "infra",
    "Research Paper": "research",
    "General AI News": "general_news",
}

_RELEASE_TYPES = {"model", "framework", "agent_platform", "infra", "research"}

_FRAMEWORK_SOURCES = {
    "LangChain Blog", "LlamaIndex Blog", "Hugging Face Blog",
    "Weights & Biases", "AWS ML Blog",
}

_FRAMEWORK_BOOST_MARGIN = 0.10


def classify_release(
    text: str,
    source: str | None = None,
) -> tuple[str, bool, float]:
    """
    Classify whether an article describes an AI release and what type.

    If the article comes from a known framework source and the classifier's
    top label is general_news or agent_platform with low confidence,
    boost the framework score to reduce misclassification.

    Returns:
        Tuple of (release_type, is_release, confidence).
    """
    if not model_registry.is_loaded:
        raise RuntimeError("Models not loaded. Call model_registry.load_all() at startup.")

    settings = get_settings()
    input_text = text[:512] if len(text) > 512 else text

    result = model_registry.classifier_pipeline(
        input_text,
        candidate_labels=settings.release_labels,
        multi_label=False,
    )

    labels = result["labels"]
    scores = result["scores"]
    label_scores = dict(zip(labels, scores))

    top_label = labels[0]
    top_score = float(scores[0])

    release_type = _LABEL_TO_RELEASE_TYPE.get(top_label, "general_news")

    if (
        source in _FRAMEWORK_SOURCES
        and release_type in ("general_news", "agent_platform")
        and top_score < 0.55
    ):
        fw_label = "Framework or Library Release and Update"
        fw_score = label_scores.get(fw_label, 0.0)
        if fw_score + _FRAMEWORK_BOOST_MARGIN >= top_score:
            release_type = "framework"
            top_score = fw_score
            logger.debug(
                "Source-boosted '%s' from %s -> framework (fw_score=%.3f)",
                source, _LABEL_TO_RELEASE_TYPE.get(top_label, top_label), fw_score,
            )

    is_release = release_type in _RELEASE_TYPES

    logger.debug(
        "Release classification: '%s' -> %s (is_release=%s, confidence=%.3f)",
        input_text[:80],
        release_type,
        is_release,
        top_score,
    )

    return release_type, is_release, top_score
