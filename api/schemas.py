"""
TechSage — Pydantic schemas for API request/response validation.
"""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class ArticleResponse(BaseModel):
    id: UUID
    title: str
    url: str
    source: str
    category: str | None
    category_confidence: float | None
    summary: str | None
    published_at: datetime | None
    created_at: datetime
    content_type: str | None = "article"
    release_type: str | None = None
    is_release: bool = False
    source_tier: int | None = None
    # Technical metadata (model releases)
    context_length_tokens: int | None = None
    model_size_params: str | None = None
    license: str | None = None
    open_source: bool | None = None
    benchmark_claims: list[dict] | None = None
    cost_changes: str | None = None
    api_changes: str | None = None
    hardware_requirements: str | None = None
    fine_tuning_supported: bool | None = None
    # Engineering impact (model/framework/agent_platform/infra releases)
    engineering_impact: dict | None = None
    # GitHub fields
    github_stars: int | None = None
    github_language: str | None = None
    github_topics: list[str] | None = None
    repo_quality_score: float | None = None
    star_velocity: float | None = None
    repo_quality_tag: str | None = None

    model_config = {"from_attributes": True}


class ArticleListResponse(BaseModel):
    articles: list[ArticleResponse]
    total: int


class ScoreBreakdown(BaseModel):
    release_priority: float
    embedding_similarity: float
    recency: float
    source_tier_weight: float
    category_affinity: float


class RankedArticleResponse(BaseModel):
    article_id: str
    title: str
    url: str
    source: str
    category: str | None
    category_confidence: float | None
    summary: str | None
    published_at: str | None
    content_type: str | None = "article"
    release_type: str | None = None
    is_release: bool = False
    source_tier: int | None = None
    # Technical metadata (model releases)
    context_length_tokens: int | None = None
    model_size_params: str | None = None
    license: str | None = None
    open_source: bool | None = None
    benchmark_claims: list[dict] | None = None
    cost_changes: str | None = None
    api_changes: str | None = None
    hardware_requirements: str | None = None
    fine_tuning_supported: bool | None = None
    # Engineering impact (model/framework/agent_platform/infra releases)
    engineering_impact: dict | None = None
    # GitHub fields
    github_stars: int | None = None
    github_language: str | None = None
    github_topics: list[str] | None = None
    repo_quality_score: float | None = None
    star_velocity: float | None = None
    repo_quality_tag: str | None = None
    score: float
    score_breakdown: ScoreBreakdown


class RankedFeedResponse(BaseModel):
    user_id: str
    articles: list[RankedArticleResponse]
    total: int


class EngineerScoreBreakdown(BaseModel):
    release_priority: float
    technical_depth: float
    recency: float
    source_tier_weight: float
    momentum: float


class EngineerArticleResponse(BaseModel):
    article_id: str
    title: str
    url: str
    source: str
    category: str | None
    category_confidence: float | None
    summary: str | None
    published_at: str | None
    content_type: str | None = "article"
    release_type: str | None = None
    is_release: bool = False
    source_tier: int | None = None
    context_length_tokens: int | None = None
    model_size_params: str | None = None
    license: str | None = None
    open_source: bool | None = None
    benchmark_claims: list[dict] | None = None
    cost_changes: str | None = None
    api_changes: str | None = None
    hardware_requirements: str | None = None
    fine_tuning_supported: bool | None = None
    engineering_impact: dict | None = None
    github_stars: int | None = None
    github_language: str | None = None
    github_topics: list[str] | None = None
    repo_quality_score: float | None = None
    star_velocity: float | None = None
    repo_quality_tag: str | None = None
    engineer_score: float
    engineer_breakdown: EngineerScoreBreakdown


class EngineerFeedResponse(BaseModel):
    user_id: str
    articles: list[EngineerArticleResponse]
    total: int


class UserCreateRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=128)
    email: str | None = None
    interests: list[str] | None = Field(None, description="Initial category interests for onboarding")


class UserResponse(BaseModel):
    id: UUID
    username: str
    email: str | None
    created_at: datetime

    model_config = {"from_attributes": True}


class UserStatsResponse(BaseModel):
    interaction_count: int
    category_preferences: dict[str, float]
    has_embedding_centroid: bool


class InteractionRequest(BaseModel):
    article_id: UUID
    interaction_type: str = Field(default="view", pattern="^(view|click|bookmark|share|like|dislike)$")


class InteractionResponse(BaseModel):
    id: UUID
    user_id: UUID
    article_id: UUID
    interaction_type: str
    created_at: datetime

    model_config = {"from_attributes": True}


class IngestionResult(BaseModel):
    fetched: int
    new_articles: int
    duplicates_url: int
    duplicates_semantic: int
    errors: int
    tier_replacements: int = 0


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    database_connected: bool
