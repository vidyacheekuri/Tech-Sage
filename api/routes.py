"""
TechSage — FastAPI route definitions.

Endpoints:
- /health — system health
- /articles — article CRUD with release_type/content_type filtering
- /users — user management
- /users/{id}/feed — personalized feed with release/content filters
- /users/{id}/interactions — interaction tracking
- /ingest — manual ingestion trigger
"""

import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import desc, func
from sqlalchemy.orm import Session

from api.schemas import (
    ArticleListResponse,
    ArticleResponse,
    ChatRequest,
    ChatResponse,
    ChatSource,
    EngineerArticleResponse,
    EngineerFeedResponse,
    EngineerScoreBreakdown,
    HealthResponse,
    IngestionResult,
    InteractionRequest,
    InteractionResponse,
    RankedArticleResponse,
    RankedFeedResponse,
    ScoreBreakdown,
    UserCreateRequest,
    UserResponse,
    UserStatsResponse,
)
from config.model_registry import model_registry
from database.models import Article, User
from database.session import get_db
from recommendation.engineer_ranking import rank_engineer_feed
from recommendation.interaction_tracker import record_interaction
from recommendation.ranking import rank_articles
from recommendation.user_profile import get_or_create_user, get_user_stats
from scheduler.ingestion_job import run_ingestion_pipeline

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["System"])
def health_check(db: Session = Depends(get_db)):
    db_ok = False
    try:
        db.execute(func.now())
        db_ok = True
    except Exception:
        pass

    return HealthResponse(
        status="healthy" if (model_registry.is_loaded and db_ok) else "degraded",
        models_loaded=model_registry.is_loaded,
        database_connected=db_ok,
    )


@router.get("/articles", response_model=ArticleListResponse, tags=["Articles"])
def list_articles(
    category: str | None = Query(None, description="Filter by topic category"),
    release_type: str | None = Query(None, description="Filter by release type"),
    content_type: str | None = Query(None, description="Filter by content type (article/github_repo)"),
    is_release: bool | None = Query(None, description="Filter releases only"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    """List articles with optional filters, ordered by newest first."""
    query = db.query(Article).filter(Article.duplicate_of.is_(None)).order_by(desc(Article.created_at))
    if category:
        query = query.filter(Article.category == category)
    if release_type:
        query = query.filter(Article.release_type == release_type)
    if content_type:
        query = query.filter(Article.content_type == content_type)
    if is_release is not None:
        query = query.filter(Article.is_release == is_release)

    total = query.count()
    articles = query.offset(offset).limit(limit).all()

    return ArticleListResponse(
        articles=[ArticleResponse.model_validate(a) for a in articles],
        total=total,
    )


@router.get("/articles/{article_id}", response_model=ArticleResponse, tags=["Articles"])
def get_article(article_id: UUID, db: Session = Depends(get_db)):
    article = db.query(Article).filter(Article.id == article_id).first()
    if not article:
        raise HTTPException(status_code=404, detail="Article not found")
    return ArticleResponse.model_validate(article)


@router.post("/articles/batch", response_model=ArticleListResponse, tags=["Articles"])
def get_articles_batch(
    article_ids: list[UUID],
    db: Session = Depends(get_db),
):
    """Fetch multiple articles by ID (used for bookmarks/saved)."""
    if not article_ids:
        return ArticleListResponse(articles=[], total=0)
    articles = db.query(Article).filter(Article.id.in_(article_ids)).all()
    return ArticleListResponse(
        articles=[ArticleResponse.model_validate(a) for a in articles],
        total=len(articles),
    )


@router.post("/chat", response_model=ChatResponse, tags=["RAG"])
def chat(req: ChatRequest, db: Session = Depends(get_db)):
    """RAG chat: ask questions about the AI ecosystem; answers are grounded in retrieved articles."""
    from rag.query import rag_query

    result = rag_query(db, req.message)
    return ChatResponse(
        answer=result["answer"],
        sources=[ChatSource(**s) for s in result["sources"]],
    )


@router.get("/articles/categories/summary", tags=["Articles"])
def get_category_summary(db: Session = Depends(get_db)):
    results = (
        db.query(Article.category, func.count(Article.id))
        .filter(Article.duplicate_of.is_(None))
        .group_by(Article.category)
        .all()
    )
    return {cat or "uncategorized": count for cat, count in results}


@router.get("/articles/releases/summary", tags=["Articles"])
def get_release_summary(
    max_age_days: int | None = Query(None, ge=1, le=365, description="Only count articles within N days"),
    db: Session = Depends(get_db),
):
    """Get count of articles per release_type, optionally limited to recent articles."""
    from datetime import datetime, timedelta, timezone

    query = (
        db.query(Article.release_type, func.count(Article.id))
        .filter(Article.duplicate_of.is_(None))
    )
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
    results = query.group_by(Article.release_type).all()
    counts = {rt or "unknown": count for rt, count in results}

    gh_query = (
        db.query(func.count(Article.id))
        .filter(Article.duplicate_of.is_(None), Article.content_type == "github_repo")
    )
    if max_age_days is not None:
        gh_query = gh_query.filter(Article.created_at >= cutoff)
    counts["github_repo"] = gh_query.scalar() or 0

    return counts


@router.post("/users", response_model=UserResponse, tags=["Users"])
def create_user(req: UserCreateRequest, db: Session = Depends(get_db)):
    from database.models import UserProfile

    user = get_or_create_user(db, req.username)
    if req.email:
        user.email = req.email
    db.flush()

    if req.interests:
        profile = db.query(UserProfile).filter(UserProfile.user_id == user.id).first()
        if not profile:
            profile = UserProfile(user_id=user.id, category_weights={}, interaction_count=0)
            db.add(profile)

        if profile.interaction_count == 0:
            weight_per = 1.0 / len(req.interests)
            profile.category_weights = {cat: weight_per for cat in req.interests}
            db.flush()

    return UserResponse.model_validate(user)


@router.get("/users/{user_id}", response_model=UserResponse, tags=["Users"])
def get_user(user_id: UUID, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return UserResponse.model_validate(user)


@router.get("/users/{user_id}/stats", response_model=UserStatsResponse, tags=["Users"])
def get_user_stats_route(user_id: UUID, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    stats = get_user_stats(db, user_id)
    return UserStatsResponse(**stats)


@router.get("/users/{user_id}/feed", response_model=RankedFeedResponse, tags=["Feed"])
def get_personalized_feed(
    user_id: UUID,
    limit: int = Query(20, ge=1, le=50),
    category: str | None = Query(None, description="Filter by topic category"),
    release_type: str | None = Query(None, description="Filter by release type"),
    content_type: str | None = Query(None, description="Filter by content type"),
    max_age_days: int | None = Query(None, ge=1, le=365, description="Only show articles published within N days"),
    db: Session = Depends(get_db),
):
    """Get personalized article feed ranked by the TechSage formula."""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    ranked = rank_articles(
        db, user_id,
        limit=limit,
        category_filter=category,
        release_type_filter=release_type,
        content_type_filter=content_type,
        max_age_days=max_age_days,
    )

    articles = [
        RankedArticleResponse(
            **{**a, "score_breakdown": ScoreBreakdown(**a["score_breakdown"])}
        )
        for a in ranked
    ]

    return RankedFeedResponse(
        user_id=str(user_id),
        articles=articles,
        total=len(articles),
    )


@router.get("/users/{user_id}/engineer-feed", response_model=EngineerFeedResponse, tags=["Engineer Feed"])
def get_engineer_feed(
    user_id: UUID,
    limit: int = Query(30, ge=1, le=100),
    release_type: str | None = Query(None, description="Filter by release type"),
    content_type: str | None = Query(None, description="Filter by content type"),
    max_age_days: int | None = Query(None, ge=1, le=365, description="Only articles within N days"),
    db: Session = Depends(get_db),
):
    """
    Engineer-focused feed ranked by technical depth, release priority, and momentum.

    No personalization — purely signal-based ranking optimized for engineers
    who want to track what matters for their stack.
    """
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    ranked = rank_engineer_feed(
        db,
        limit=limit,
        release_type_filter=release_type,
        content_type_filter=content_type,
        max_age_days=max_age_days,
    )

    articles = [
        EngineerArticleResponse(
            **{**a, "engineer_breakdown": EngineerScoreBreakdown(**a["engineer_breakdown"])}
        )
        for a in ranked
    ]

    return EngineerFeedResponse(
        user_id=str(user_id),
        articles=articles,
        total=len(articles),
    )


@router.post(
    "/users/{user_id}/interactions",
    response_model=InteractionResponse | dict,
    tags=["Interactions"],
)
def create_interaction(
    user_id: UUID,
    req: InteractionRequest,
    db: Session = Depends(get_db),
):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    article = db.query(Article).filter(Article.id == req.article_id).first()
    if not article:
        raise HTTPException(status_code=404, detail="Article not found")

    interaction = record_interaction(
        db, user_id, req.article_id, req.interaction_type
    )

    if interaction is None:
        return {"message": "Interaction already recorded."}

    return InteractionResponse.model_validate(interaction)


@router.post("/ingest", response_model=IngestionResult, tags=["Ingestion"])
async def trigger_ingestion(db: Session = Depends(get_db)):
    """Manually trigger the news ingestion pipeline."""
    result = await run_ingestion_pipeline(db)
    return IngestionResult(**result)


@router.post("/articles/resummarize", tags=["Ingestion"])
def resummarize_articles(db: Session = Depends(get_db)):
    """Re-summarize all existing articles with the current (longer) settings."""
    from news_ingestion.summarizer import summarize_article

    articles = db.query(Article).filter(
        Article.cleaned_content.isnot(None),
        func.length(Article.cleaned_content) > 100
    ).all()

    updated = 0
    errors = 0
    for article in articles:
        try:
            new_summary = summarize_article(article.cleaned_content)
            if new_summary and len(new_summary) > len(article.summary or ""):
                article.summary = new_summary
                updated += 1
                if updated % 5 == 0:
                    db.commit()
        except Exception as e:
            errors += 1
            db.rollback()
            logger.warning("Failed to re-summarize article %s: %s", article.id, e)
    try:
        db.commit()
    except Exception:
        db.rollback()
    return {"total": len(articles), "updated": updated, "errors": errors}
