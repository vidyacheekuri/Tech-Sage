"""
Article content cleaner.

Converts raw HTML/RSS content into clean plaintext suitable for:
- Embedding generation (needs semantic content, not markup)
- Zero-shot classification (needs readable sentences)
- Summarization (needs coherent text)

Pipeline: HTML strip -> whitespace normalize -> encoding fix -> length check
"""

import html
import logging
import re

from bs4 import BeautifulSoup

from news_ingestion.fetcher import RawArticle

logger = logging.getLogger(__name__)

# Minimum cleaned content length to be useful for ML processing
_MIN_CONTENT_LENGTH = 50


def clean_html(raw_html: str) -> str:
    """Strip HTML tags and decode entities."""
    if not raw_html:
        return ""

    soup = BeautifulSoup(raw_html, "html.parser")

    # Remove script and style elements
    for element in soup(["script", "style", "nav", "footer", "header", "aside"]):
        element.decompose()

    text = soup.get_text(separator=" ")

    # Decode HTML entities
    text = html.unescape(text)

    return text


def normalize_whitespace(text: str) -> str:
    """Collapse multiple whitespace characters into single spaces."""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def remove_boilerplate(text: str) -> str:
    """Remove common boilerplate patterns from news articles."""
    patterns = [
        r"Sign up for.*?newsletter.*?\.",
        r"Subscribe to.*?\.",
        r"Read more:.*",
        r"Click here to.*?\.",
        r"Advertisement\s*",
        r"Sponsored\s*",
        r"\[.*?chars?\]",  # NewsAPI truncation markers like [+1234 chars]
        r"…\s*\[\+\d+\s*chars?\]",
    ]
    for pattern in patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    return text.strip()


def clean_article(raw_article: RawArticle) -> str:
    """
    Clean a raw article and return text suitable for ML processing.

    Combines title with the best available content (content > description)
    and applies full cleaning pipeline.

    Args:
        raw_article: RawArticle from the fetcher.

    Returns:
        Cleaned text string (title + content).
    """
    # Pick the best content source
    content = raw_article.content or raw_article.description or ""

    # Clean HTML
    content = clean_html(content)

    # Normalize
    content = normalize_whitespace(content)
    content = remove_boilerplate(content)

    # Combine title with content for richer semantic representation
    title = normalize_whitespace(raw_article.title)
    combined = f"{title}. {content}" if content else title

    if len(combined) < _MIN_CONTENT_LENGTH:
        logger.debug(
            "Article '%s' has insufficient content (%d chars) after cleaning.",
            raw_article.title[:60],
            len(combined),
        )

    return combined
