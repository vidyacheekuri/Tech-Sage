"""
TechSage — AI ecosystem source fetcher.

Tiered source architecture:
  Tier 1: Official AI lab blogs (highest signal)
  Tier 2: Framework / infrastructure blogs
  Tier 3: AI news sections of major publications
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

import feedparser
import httpx

logger = logging.getLogger(__name__)

# Source -> (URL, Tier)
AI_FEEDS: dict[str, tuple[str, int]] = {
    # ── Tier 1: Official AI Labs ──
    "OpenAI Blog": ("https://openai.com/blog/rss.xml", 1),
    "Google AI Blog": ("https://blog.research.google/feeds/posts/default?alt=rss", 1),
    "Google DeepMind": ("https://deepmind.google/blog/rss.xml", 1),
    "Anthropic News": (
        "https://raw.githubusercontent.com/Olshansk/rss-feeds/main/feeds/feed_anthropic_news.xml",
        1,
    ),
    "Anthropic Research": (
        "https://raw.githubusercontent.com/Olshansk/rss-feeds/main/feeds/feed_anthropic_research.xml",
        1,
    ),
    "Meta AI Research": ("https://research.facebook.com/feed/", 1),
    "NVIDIA AI Blog": ("https://developer.nvidia.com/blog/feed", 1),
    # ── Tier 2: Framework / Infra Blogs ──
    "LangChain Blog": ("https://blog.langchain.dev/rss/", 2),
    "AWS ML Blog": ("https://aws.amazon.com/blogs/machine-learning/feed/", 2),
    "Hugging Face Blog": ("https://huggingface.co/blog/feed.xml", 2),
    "Weights & Biases": ("https://wandb.ai/fully-connected/rss.xml", 2),
    "Meta Engineering": ("https://engineering.fb.com/feed/", 2),
    "ArXiv AI+ML": ("https://rss.arxiv.org/rss/cs.AI+cs.LG", 2),
    # ── Tier 3: AI News Sections ──
    "TechCrunch AI": ("https://techcrunch.com/category/artificial-intelligence/feed/", 3),
    "VentureBeat AI": ("https://venturebeat.com/category/ai/feed/", 3),
    "The Verge AI": ("https://www.theverge.com/rss/ai-artificial-intelligence/index.xml", 3),
    "Ars Technica AI": ("https://feeds.arstechnica.com/arstechnica/technology-lab", 3),
    "MIT Tech Review AI": ("https://www.technologyreview.com/feed/", 3),
    "Hacker News AI": ("https://hnrss.org/newest?q=AI+OR+LLM+OR+GPT&points=50", 3),
}

_TIMEOUT = httpx.Timeout(15.0, connect=10.0)


@dataclass
class RawArticle:
    """Unified representation of a fetched article before cleaning."""

    title: str
    url: str
    source: str
    source_tier: int = 3
    author: str | None = None
    published_at: datetime | None = None
    content: str = ""
    description: str = ""
    extra: dict = field(default_factory=dict)


async def fetch_rss_feeds() -> list[RawArticle]:
    """Fetch articles from all configured AI-focused RSS feeds."""
    articles = []

    async with httpx.AsyncClient(timeout=_TIMEOUT, follow_redirects=True) as client:
        for source_name, (feed_url, tier) in AI_FEEDS.items():
            try:
                response = await client.get(feed_url)
                response.raise_for_status()
                feed = feedparser.parse(response.text)

                count = 0
                for entry in feed.entries[:20]:
                    published = None
                    if hasattr(entry, "published_parsed") and entry.published_parsed:
                        try:
                            published = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
                        except (TypeError, ValueError):
                            pass

                    articles.append(
                        RawArticle(
                            title=entry.get("title", "").strip(),
                            url=entry.get("link", "").strip(),
                            source=source_name,
                            source_tier=tier,
                            author=entry.get("author"),
                            published_at=published,
                            content=entry.get("content", [{}])[0].get("value", "")
                            if entry.get("content")
                            else "",
                            description=entry.get("summary", ""),
                        )
                    )
                    count += 1

                logger.info("Fetched %d articles from %s (Tier %d)", count, source_name, tier)

            except httpx.HTTPStatusError as e:
                logger.warning("HTTP error fetching %s: %s", source_name, e)
            except Exception as e:
                logger.error("Failed to fetch %s: %s", source_name, e)

    return articles


async def fetch_all_sources() -> list[RawArticle]:
    """Aggregate articles from all AI-focused sources."""
    articles = await fetch_rss_feeds()
    valid = [a for a in articles if a.title and a.url]
    logger.info("Total fetched: %d articles (%d valid) from AI sources.", len(articles), len(valid))
    return valid
