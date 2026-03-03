"""
TechSage — GitHub trending AI repository ingestion with quality scoring.

Fetches trending repositories from GitHub's search API filtered by AI-related topics.
Runs on a separate schedule (every 6 hours).

Quality score formula:
    repo_quality_score = log(stars + 1) + 0.5 * log(forks + 1)
                         + 0.3 * contributor_count + star_velocity_score

Star velocity: stars gained per day since creation (capped at repo age of 1 day min).

Quality tags (assigned after all repos are scored):
    high_quality  — top 25% by score
    experimental  — middle 50%
    early_stage   — bottom 25%
"""

import logging
import math
from datetime import datetime, timedelta, timezone

import httpx

from config.settings import get_settings
from news_ingestion.fetcher import RawArticle

logger = logging.getLogger(__name__)

_TIMEOUT = httpx.Timeout(20.0, connect=10.0)

SEARCH_TOPICS = [
    "ai", "llm", "agent", "genai", "machine-learning",
    "deep-learning", "transformers", "rag", "langchain", "mlops",
]

_LOOKBACK_DAYS = 7


def compute_star_velocity(stars: int, created_at: datetime | None) -> float:
    """Stars per day since repo creation. Minimum age is 1 day to avoid division by zero."""
    if not created_at:
        return 0.0
    now = datetime.now(timezone.utc)
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    age_days = max(1.0, (now - created_at).total_seconds() / 86400.0)
    return stars / age_days


def compute_quality_score(
    stars: int, forks: int, contributor_count: int, star_velocity: float
) -> float:
    """
    repo_quality_score = log(stars + 1) + 0.5 * log(forks + 1)
                         + 0.3 * contributor_count + star_velocity_score

    star_velocity_score is log(velocity + 1) to keep it on the same scale.
    """
    return (
        math.log(stars + 1)
        + 0.5 * math.log(forks + 1)
        + 0.3 * min(contributor_count, 100)  # cap to avoid outsized influence
        + math.log(star_velocity + 1)
    )


def assign_quality_tags(repos: list[dict]) -> list[dict]:
    """
    Assign quality tags based on percentile rank of repo_quality_score.
    Mutates the dicts in-place and returns the list.
    """
    if not repos:
        return repos

    sorted_repos = sorted(repos, key=lambda r: r.get("repo_quality_score", 0))
    n = len(sorted_repos)
    p25 = max(1, n // 4)
    p75 = max(p25 + 1, n * 3 // 4)

    for i, repo in enumerate(sorted_repos):
        if i < p25:
            repo["repo_quality_tag"] = "early_stage"
        elif i >= p75:
            repo["repo_quality_tag"] = "high_quality"
        else:
            repo["repo_quality_tag"] = "experimental"

    return repos


async def _fetch_contributor_count(
    client: httpx.AsyncClient, full_name: str
) -> int:
    """Fetch contributor count for a repo. Returns 0 on failure."""
    try:
        resp = await client.get(
            f"https://api.github.com/repos/{full_name}/contributors",
            params={"per_page": 1, "anon": "false"},
        )
        if resp.status_code == 200:
            # GitHub returns contributor count in the Link header's last page number
            link = resp.headers.get("link", "")
            if 'rel="last"' in link:
                import re
                match = re.search(r'page=(\d+)>;\s*rel="last"', link)
                if match:
                    return int(match.group(1))
            return len(resp.json())
        return 0
    except Exception:
        return 0


async def fetch_trending_repos() -> list[RawArticle]:
    """
    Fetch trending AI repositories with quality scoring.
    Returns RawArticle list with quality metrics in the extra dict.
    """
    settings = get_settings()
    articles = []
    seen_urls = set()
    quality_data = []

    headers = {"Accept": "application/vnd.github+json"}
    if settings.github_token:
        headers["Authorization"] = f"Bearer {settings.github_token}"

    since_date = (datetime.now(timezone.utc) - timedelta(days=_LOOKBACK_DAYS)).strftime("%Y-%m-%d")

    import asyncio

    searches: list[tuple[str, str, str, int]] = []
    for topic in SEARCH_TOPICS:
        searches.append((topic, f"topic:{topic} pushed:>{since_date} stars:>50", "stars", 15))
    for topic in SEARCH_TOPICS[:5]:
        searches.append((topic, f"topic:{topic} created:>{since_date} stars:>10", "stars", 10))
    for topic in SEARCH_TOPICS[:5]:
        searches.append((topic, f"topic:{topic} pushed:>{since_date} stars:>20", "updated", 10))

    async with httpx.AsyncClient(timeout=_TIMEOUT, headers=headers) as client:
        for idx, (topic, query, sort_by, per_page) in enumerate(searches):
            if idx > 0:
                await asyncio.sleep(6.5 if not settings.github_token else 1.5)
            try:
                response = await client.get(
                    "https://api.github.com/search/repositories",
                    params={
                        "q": query,
                        "sort": sort_by,
                        "order": "desc",
                        "per_page": per_page,
                    },
                )
                response.raise_for_status()
                data = response.json()

                for repo in data.get("items", []):
                    url = repo.get("html_url", "")
                    if url in seen_urls:
                        continue
                    seen_urls.add(url)

                    created = None
                    if repo.get("created_at"):
                        try:
                            created = datetime.fromisoformat(repo["created_at"].replace("Z", "+00:00"))
                        except ValueError:
                            pass

                    description = repo.get("description") or ""
                    name = repo.get("full_name", "")
                    stars = repo.get("stargazers_count", 0)
                    forks = repo.get("forks_count", 0)
                    language = repo.get("language") or ""
                    topics = repo.get("topics", [])

                    contributor_count = await _fetch_contributor_count(client, name)

                    velocity = compute_star_velocity(stars, created)
                    quality_score = compute_quality_score(stars, forks, contributor_count, velocity)

                    content_parts = [
                        f"{name}: {description}",
                        f"Language: {language}" if language else "",
                        f"Stars: {stars}, Forks: {forks}",
                        f"Topics: {', '.join(topics)}" if topics else "",
                    ]
                    content = ". ".join(p for p in content_parts if p)

                    idx = len(articles)
                    articles.append(
                        RawArticle(
                            title=f"{name} — {description[:120]}" if description else name,
                            url=url,
                            source="GitHub Trending",
                            source_tier=4,
                            published_at=created,
                            content=content,
                            description=description,
                            extra={
                                "github_stars": stars,
                                "github_forks": forks,
                                "github_language": language,
                                "github_topics": topics,
                                "github_contributors": contributor_count,
                                "content_type": "github_repo",
                                "repo_quality_score": round(quality_score, 4),
                                "star_velocity": round(velocity, 4),
                            },
                        )
                    )
                    quality_data.append({
                        "idx": idx,
                        "repo_quality_score": quality_score,
                    })

                logger.info("Fetched %d repos for topic '%s'", len(data.get("items", [])), topic)

            except httpx.HTTPStatusError as e:
                logger.warning("GitHub API error for topic '%s': %s", topic, e)
            except Exception as e:
                logger.error("GitHub fetch failed for topic '%s': %s", topic, e)

    # Assign quality tags based on percentile ranking
    assign_quality_tags(quality_data)
    for qd in quality_data:
        articles[qd["idx"]].extra["repo_quality_tag"] = qd["repo_quality_tag"]

    logger.info("Total GitHub trending repos fetched: %d (deduplicated, scored)", len(articles))
    return articles
