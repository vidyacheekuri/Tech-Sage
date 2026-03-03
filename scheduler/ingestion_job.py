"""
TechSage — Ingestion pipeline orchestrator and scheduler.

Two pipelines:
1. AI News Pipeline (runs every 5 min):
   Fetch RSS -> Clean -> Embed -> Deduplicate (tier-aware) -> Classify (topic) ->
   Classify (release) -> Summarize -> Store

2. GitHub Trending Pipeline (runs every 6 hours):
   Fetch trending repos -> Clean -> Embed -> Deduplicate -> Classify (release) ->
   Summarize -> Store with content_type='github_repo'
"""

import logging

from apscheduler.schedulers.background import BackgroundScheduler
from sqlalchemy.orm import Session

from config.settings import get_settings
from database.models import Article
from database.session import get_session_factory
from news_ingestion.classifier import classify_article
from news_ingestion.cleaner import clean_article
from news_ingestion.deduplication import (
    handle_tiered_duplicate,
    is_duplicate,
    is_duplicate_url,
    mark_as_duplicate,
)
from news_ingestion.embedding import generate_embedding
from news_ingestion.fetcher import fetch_all_sources
from news_ingestion.github_trending import fetch_trending_repos
from news_ingestion.impact_analyzer import analyze_engineering_impact
from news_ingestion.release_classifier import classify_release
from news_ingestion.summarizer import summarize_article
from news_ingestion.technical_extractor import extract_model_metadata

logger = logging.getLogger(__name__)

_scheduler: BackgroundScheduler | None = None


async def run_ingestion_pipeline(db: Session) -> dict:
    """Execute the full AI news ingestion pipeline."""
    stats = {
        "fetched": 0,
        "new_articles": 0,
        "duplicates_url": 0,
        "duplicates_semantic": 0,
        "tier_replacements": 0,
        "errors": 0,
    }

    logger.info("Starting AI news ingestion pipeline...")
    raw_articles = await fetch_all_sources()
    stats["fetched"] = len(raw_articles)
    logger.info("Fetched %d raw articles from AI sources.", len(raw_articles))

    for raw_article in raw_articles:
        try:
            if is_duplicate_url(raw_article.url, db):
                stats["duplicates_url"] += 1
                continue

            cleaned_text = clean_article(raw_article)
            if len(cleaned_text) < 50:
                stats["errors"] += 1
                continue

            embedding = generate_embedding(cleaned_text)

            is_dup, sim_score, dup_id = is_duplicate(embedding, db)
            if is_dup:
                action = handle_tiered_duplicate(raw_article.source_tier, dup_id, db)
                if action == "skip":
                    stats["duplicates_semantic"] += 1
                    continue
                # action == "replace": new article is higher tier, proceed and mark old as dup

            # Topic classification
            category, confidence = classify_article(cleaned_text)

            # Release classification (secondary pass — source hint for framework blogs)
            release_type, is_release, _ = classify_release(cleaned_text, source=raw_article.source)

            summary = summarize_article(cleaned_text)

            # Technical metadata extraction (model releases only)
            tech_meta = {}
            if release_type == "model":
                meta = extract_model_metadata(raw_article.title, cleaned_text)
                tech_meta = {
                    "context_length_tokens": meta.context_length_tokens,
                    "model_size_params": meta.model_size_params,
                    "license": meta.license,
                    "open_source": meta.open_source,
                    "benchmark_claims": meta.benchmark_claims,
                    "cost_changes": meta.cost_changes,
                    "api_changes": meta.api_changes,
                    "hardware_requirements": meta.hardware_requirements,
                    "fine_tuning_supported": meta.fine_tuning_supported,
                }

            # Engineering impact assessment (model/framework/agent_platform/infra)
            eng_impact = analyze_engineering_impact(
                raw_article.title, cleaned_text, release_type
            )

            article = Article(
                title=raw_article.title,
                url=raw_article.url,
                source=raw_article.source,
                source_tier=raw_article.source_tier,
                author=raw_article.author,
                published_at=raw_article.published_at,
                raw_content=raw_article.content or raw_article.description,
                cleaned_content=cleaned_text,
                summary=summary,
                category=category,
                category_confidence=confidence,
                content_type="article",
                release_type=release_type,
                is_release=is_release,
                embedding=embedding.tolist(),
                engineering_impact=eng_impact,
                **tech_meta,
            )
            db.add(article)
            db.flush()

            # If this replaced a lower-tier duplicate, mark the old one
            if is_dup and dup_id:
                mark_as_duplicate(dup_id, str(article.id), db)
                stats["tier_replacements"] += 1

            stats["new_articles"] += 1
            logger.info(
                "Ingested: '%s' [%s | %s | Tier%d] — %s",
                raw_article.title[:60],
                category,
                release_type,
                raw_article.source_tier,
                raw_article.source,
            )

            # Commit every 5 articles so they become visible to the frontend immediately
            if stats["new_articles"] % 5 == 0:
                db.commit()

        except Exception as e:
            stats["errors"] += 1
            logger.error(
                "Error processing article '%s': %s",
                raw_article.title[:60] if raw_article.title else "unknown",
                e,
                exc_info=True,
            )

    db.commit()
    logger.info(
        "News ingestion complete: %d fetched, %d new, %d URL dups, %d semantic dups, %d tier replacements, %d errors.",
        stats["fetched"], stats["new_articles"], stats["duplicates_url"],
        stats["duplicates_semantic"], stats["tier_replacements"], stats["errors"],
    )
    return stats


async def run_github_pipeline(db: Session) -> dict:
    """Execute the GitHub trending repository ingestion pipeline."""
    stats = {"fetched": 0, "new_repos": 0, "duplicates": 0, "errors": 0}

    logger.info("Starting GitHub trending ingestion pipeline...")
    raw_repos = await fetch_trending_repos()
    stats["fetched"] = len(raw_repos)

    for raw_repo in raw_repos:
        try:
            if is_duplicate_url(raw_repo.url, db):
                stats["duplicates"] += 1
                continue

            cleaned_text = clean_article(raw_repo)
            if len(cleaned_text) < 20:
                stats["errors"] += 1
                continue

            embedding = generate_embedding(cleaned_text)

            is_dup, _, _ = is_duplicate(embedding, db)
            if is_dup:
                stats["duplicates"] += 1
                continue

            release_type, is_release, _ = classify_release(cleaned_text, source=raw_repo.source)
            # GitHub repos default to github_repo release type unless clearly something else
            if release_type == "general_news":
                release_type = "github_repo"

            summary = summarize_article(cleaned_text)

            article = Article(
                title=raw_repo.title,
                url=raw_repo.url,
                source=raw_repo.source,
                source_tier=4,
                published_at=raw_repo.published_at,
                raw_content=raw_repo.content,
                cleaned_content=cleaned_text,
                summary=summary,
                category="AI Frameworks & Libraries",
                content_type="github_repo",
                release_type=release_type,
                is_release=is_release,
                embedding=embedding.tolist(),
                github_stars=raw_repo.extra.get("github_stars"),
                github_language=raw_repo.extra.get("github_language"),
                github_topics=raw_repo.extra.get("github_topics"),
                repo_quality_score=raw_repo.extra.get("repo_quality_score"),
                star_velocity=raw_repo.extra.get("star_velocity"),
                repo_quality_tag=raw_repo.extra.get("repo_quality_tag"),
            )
            db.add(article)
            db.flush()

            stats["new_repos"] += 1
            logger.info(
                "Ingested repo: '%s' [%s, %d stars]",
                raw_repo.title[:60],
                release_type,
                raw_repo.extra.get("github_stars", 0),
            )

            if stats["new_repos"] % 5 == 0:
                db.commit()

        except Exception as e:
            stats["errors"] += 1
            logger.error("Error processing repo '%s': %s", raw_repo.title[:60], e, exc_info=True)

    db.commit()
    logger.info(
        "GitHub ingestion complete: %d fetched, %d new, %d dups, %d errors.",
        stats["fetched"], stats["new_repos"], stats["duplicates"], stats["errors"],
    )
    return stats


def _scheduled_news_job() -> None:
    """Wrapper for the news ingestion scheduler."""
    import asyncio

    logger.info("Scheduled news ingestion triggered.")
    factory = get_session_factory()
    session = factory()
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(run_ingestion_pipeline(session))
        session.commit()
        logger.info("Scheduled news ingestion result: %s", result)
    except Exception as e:
        session.rollback()
        logger.error("Scheduled news ingestion failed: %s", e, exc_info=True)
    finally:
        session.close()
        loop.close()


def _scheduled_github_job() -> None:
    """Wrapper for the GitHub trending scheduler."""
    import asyncio

    logger.info("Scheduled GitHub ingestion triggered.")
    factory = get_session_factory()
    session = factory()
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(run_github_pipeline(session))
        session.commit()
        logger.info("Scheduled GitHub ingestion result: %s", result)
    except Exception as e:
        session.rollback()
        logger.error("Scheduled GitHub ingestion failed: %s", e, exc_info=True)
    finally:
        session.close()
        loop.close()


def start_scheduler() -> None:
    """Start both background ingestion schedulers."""
    global _scheduler
    settings = get_settings()

    if _scheduler is not None:
        logger.warning("Scheduler already running.")
        return

    _scheduler = BackgroundScheduler()

    # News ingestion — every N minutes
    _scheduler.add_job(
        _scheduled_news_job,
        "interval",
        minutes=settings.ingestion_interval_minutes,
        id="news_ingestion_job",
        name="AI News Ingestion",
        max_instances=1,
        coalesce=True,
        next_run_time=None,
    )

    # GitHub trending — every N hours
    _scheduler.add_job(
        _scheduled_github_job,
        "interval",
        hours=settings.github_interval_hours,
        id="github_ingestion_job",
        name="GitHub Trending Ingestion",
        max_instances=1,
        coalesce=True,
        next_run_time=None,
    )

    # Immediate first runs
    _scheduler.add_job(
        _scheduled_news_job,
        id="initial_news_ingestion",
        name="Initial News Ingestion",
        max_instances=1,
    )
    _scheduler.add_job(
        _scheduled_github_job,
        id="initial_github_ingestion",
        name="Initial GitHub Ingestion",
        max_instances=1,
    )

    _scheduler.start()
    logger.info(
        "TechSage scheduler started — news every %d min, GitHub every %d hours.",
        settings.ingestion_interval_minutes,
        settings.github_interval_hours,
    )


def stop_scheduler() -> None:
    """Stop the background ingestion scheduler."""
    global _scheduler
    if _scheduler:
        _scheduler.shutdown(wait=False)
        _scheduler = None
        logger.info("Ingestion scheduler stopped.")
