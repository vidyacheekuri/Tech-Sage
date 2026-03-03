"""
TechSage — Engineer-focused ranking engine.

Separate from the main personalized feed. Optimized for engineering signal:
technical depth, release importance, and momentum over personalization.

EngineerScore =
    0.35 * ReleasePriority
  + 0.20 * TechnicalDepthScore
  + 0.20 * RecencyScore
  + 0.15 * SourceTierWeight
  + 0.10 * MomentumScore

TechnicalDepthScore (0-1):
    1.0 — benchmark_claims populated (quantitative evaluation data)
    0.8 — api_changes populated (concrete integration impact)
    0.7 — engineering_impact with migration_risk=high (actionable infra change)
    0.6 — model_size_params or context_length_tokens populated
    0.4 — has summary but no structured metadata
    0.2 — general_news with no technical depth

MomentumScore (0-1):
    Combines star velocity (GitHub), duplicate coverage count (cross-source signal),
    and a recency boost for articles < 12 hours old.
"""

import logging
import math
from datetime import datetime, timedelta, timezone
from uuid import UUID

from sqlalchemy import desc, func
from sqlalchemy.orm import Session

from database.models import Article
from recommendation.ranking import (
    RELEASE_PRIORITY,
    SOURCE_TIER_WEIGHT,
    compute_recency_score,
    compute_release_priority,
    compute_source_tier_weight,
)

logger = logging.getLogger(__name__)

_W_RELEASE = 0.35
_W_TECH_DEPTH = 0.20
_W_RECENCY = 0.20
_W_SOURCE_TIER = 0.15
_W_MOMENTUM = 0.10

_MOMENTUM_RECENCY_BOOST_HOURS = 12.0


def compute_technical_depth(article: Article) -> float:
    if article.benchmark_claims:
        return 1.0

    if article.api_changes:
        return 0.8

    impact = article.engineering_impact
    if impact and isinstance(impact, dict):
        if impact.get("migration_risk") == "high":
            return 0.7

    if article.model_size_params or article.context_length_tokens:
        return 0.6

    if article.summary and len(article.summary) > 30:
        return 0.4

    return 0.2


def _compute_coverage_count(article_id: str, db: Session) -> int:
    """Count how many articles are marked as duplicates of this one (cross-source coverage)."""
    count = (
        db.query(func.count(Article.id))
        .filter(Article.duplicate_of == article_id)
        .scalar()
    )
    return count or 0


def compute_momentum(
    article: Article,
    coverage_count: int,
) -> float:
    """
    Momentum = normalized(star_velocity_component + coverage_component + recency_burst)

    Each sub-component is in [0, 1], combined and re-normalized.
    """
    # Star velocity component (GitHub repos only)
    velocity_score = 0.0
    if article.star_velocity and article.star_velocity > 0:
        velocity_score = min(1.0, math.log(article.star_velocity + 1) / 6.0)

    # Coverage count: more sources covering the same release = higher momentum
    coverage_score = min(1.0, coverage_count / 5.0)

    # Recency burst: articles < 12 hours old get a boost
    recency_burst = 0.0
    pub = article.published_at or article.created_at
    if pub:
        if pub.tzinfo is None:
            pub = pub.replace(tzinfo=timezone.utc)
        hours_old = max(0, (datetime.now(timezone.utc) - pub).total_seconds() / 3600.0)
        if hours_old < _MOMENTUM_RECENCY_BOOST_HOURS:
            recency_burst = 1.0 - (hours_old / _MOMENTUM_RECENCY_BOOST_HOURS)

    raw = 0.4 * velocity_score + 0.35 * coverage_score + 0.25 * recency_burst
    return min(1.0, raw)


def _build_article_dict(article: Article, score: float, breakdown: dict) -> dict:
    return {
        "article_id": str(article.id),
        "title": article.title,
        "url": article.url,
        "source": article.source,
        "category": article.category,
        "category_confidence": article.category_confidence,
        "summary": article.summary,
        "published_at": article.published_at.isoformat() if article.published_at else None,
        "created_at": article.created_at.isoformat() if article.created_at else None,
        "content_type": article.content_type,
        "release_type": article.release_type,
        "is_release": article.is_release,
        "source_tier": article.source_tier,
        "context_length_tokens": article.context_length_tokens,
        "model_size_params": article.model_size_params,
        "license": article.license,
        "open_source": article.open_source,
        "benchmark_claims": article.benchmark_claims,
        "cost_changes": article.cost_changes,
        "api_changes": article.api_changes,
        "hardware_requirements": article.hardware_requirements,
        "fine_tuning_supported": article.fine_tuning_supported,
        "engineering_impact": article.engineering_impact,
        "github_stars": article.github_stars,
        "github_language": article.github_language,
        "github_topics": article.github_topics,
        "repo_quality_score": article.repo_quality_score,
        "star_velocity": article.star_velocity,
        "repo_quality_tag": article.repo_quality_tag,
        "engineer_score": round(score, 4),
        "engineer_breakdown": breakdown,
    }


def rank_engineer_feed(
    db: Session,
    limit: int = 30,
    release_type_filter: str | None = None,
    content_type_filter: str | None = None,
    max_age_days: int | None = None,
) -> list[dict]:
    """
    Rank articles using the engineer-focused formula.

    No user personalization — purely signal-based ranking.
    """
    query = (
        db.query(Article)
        .filter(Article.duplicate_of.is_(None))
        .order_by(desc(Article.created_at))
    )
    if release_type_filter:
        query = query.filter(Article.release_type == release_type_filter)
    if content_type_filter:
        query = query.filter(Article.content_type == content_type_filter)
    if max_age_days is not None:
        from sqlalchemy import case
        cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
        effective_date = case(
            (Article.content_type == "github_repo", Article.created_at),
            else_=Article.published_at,
        )
        query = query.filter(
            (effective_date >= cutoff) | (effective_date.is_(None))
        )

    candidates = query.limit(400).all()

    if not candidates:
        logger.info("No candidates for engineer feed.")
        return []

    scored = []
    for article in candidates:
        release_pri = compute_release_priority(article.release_type)
        tech_depth = compute_technical_depth(article)
        recency = compute_recency_score(article.published_at or article.created_at)
        tier_weight = compute_source_tier_weight(article.source_tier)
        coverage = _compute_coverage_count(str(article.id), db)
        momentum = compute_momentum(article, coverage)

        score = (
            _W_RELEASE * release_pri
            + _W_TECH_DEPTH * tech_depth
            + _W_RECENCY * recency
            + _W_SOURCE_TIER * tier_weight
            + _W_MOMENTUM * momentum
        )

        breakdown = {
            "release_priority": round(release_pri, 4),
            "technical_depth": round(tech_depth, 4),
            "recency": round(recency, 4),
            "source_tier_weight": round(tier_weight, 4),
            "momentum": round(momentum, 4),
        }

        scored.append(_build_article_dict(article, score, breakdown))

    scored.sort(key=lambda x: x["engineer_score"], reverse=True)

    logger.info(
        "Engineer feed: ranked %d articles (top score=%.4f).",
        len(scored),
        scored[0]["engineer_score"] if scored else 0,
    )

    return scored[:limit]
