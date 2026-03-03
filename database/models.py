"""
TechSage Database ORM models.

Schema:
- Article: stores AI ecosystem content with release detection metadata
- User / UserProfile / UserInteraction: personalization layer (unchanged)

Key additions over TechSage:
- content_type: 'article' or 'github_repo'
- is_release: boolean flag for detected releases
- release_type: model, framework, agent_platform, infra, research, github_repo, general_news
- source_tier: 1=Official lab, 2=Framework/tool, 3=Media, 4=GitHub
- duplicate_of: links semantic duplicates to the preferred (Tier1) source
"""

import uuid
from datetime import datetime, timezone

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class Article(Base):
    __tablename__ = "articles"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String(512), nullable=False)
    url = Column(String(2048), nullable=False, unique=True)
    source = Column(String(256), nullable=False)
    author = Column(String(256), nullable=True)
    published_at = Column(DateTime(timezone=True), nullable=True)
    raw_content = Column(Text, nullable=True)
    cleaned_content = Column(Text, nullable=False)
    summary = Column(Text, nullable=True)
    category = Column(String(64), nullable=True, index=True)
    category_confidence = Column(Float, nullable=True)

    # TechSage release intelligence fields
    content_type = Column(String(32), nullable=False, default="article")
    is_release = Column(Boolean, nullable=False, default=False)
    release_type = Column(String(32), nullable=True)
    source_tier = Column(Integer, nullable=True)
    duplicate_of = Column(UUID(as_uuid=True), ForeignKey("articles.id"), nullable=True)

    # Technical metadata (populated only for release_type='model')
    context_length_tokens = Column(Integer, nullable=True)
    model_size_params = Column(Text, nullable=True)
    license = Column(Text, nullable=True)
    open_source = Column(Boolean, nullable=True)
    benchmark_claims = Column(JSONB, nullable=True)
    cost_changes = Column(Text, nullable=True)
    api_changes = Column(Text, nullable=True)
    hardware_requirements = Column(Text, nullable=True)
    fine_tuning_supported = Column(Boolean, nullable=True)

    # Engineering impact assessment (model/framework/agent_platform/infra releases)
    engineering_impact = Column(JSONB, nullable=True)

    # GitHub-specific fields (null for articles)
    github_stars = Column(Integer, nullable=True)
    github_language = Column(String(64), nullable=True)
    github_topics = Column(JSONB, nullable=True)
    repo_quality_score = Column(Float, nullable=True)
    star_velocity = Column(Float, nullable=True)
    repo_quality_tag = Column(String(32), nullable=True)

    # 384-dimensional embedding from all-MiniLM-L6-v2
    embedding = Column(Vector(384), nullable=True)

    created_at = Column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True
    )
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    interactions = relationship("UserInteraction", back_populates="article")

    __table_args__ = (
        Index("ix_articles_created_at_desc", created_at.desc()),
        Index("ix_articles_category_created", category, created_at.desc()),
        Index("ix_articles_release_type", release_type, created_at.desc()),
        Index("ix_articles_content_type", content_type, created_at.desc()),
    )

    def __repr__(self) -> str:
        return f"<Article(id={self.id}, title='{self.title[:50]}...', release_type={self.release_type})>"


class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = Column(String(128), nullable=False, unique=True)
    email = Column(String(256), nullable=True)
    created_at = Column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    profile = relationship("UserProfile", back_populates="user", uselist=False)
    interactions = relationship("UserInteraction", back_populates="user")

    def __repr__(self) -> str:
        return f"<User(id={self.id}, username='{self.username}')>"


class UserProfile(Base):
    __tablename__ = "user_profiles"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(
        UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, unique=True
    )
    category_weights = Column(JSONB, default=dict)
    embedding_centroid = Column(Vector(384), nullable=True)
    interaction_count = Column(Integer, default=0)
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    user = relationship("User", back_populates="profile")

    def __repr__(self) -> str:
        return f"<UserProfile(user_id={self.user_id}, interactions={self.interaction_count})>"


class UserInteraction(Base):
    __tablename__ = "user_interactions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    article_id = Column(
        UUID(as_uuid=True), ForeignKey("articles.id"), nullable=False
    )
    interaction_type = Column(String(32), nullable=False, default="view")
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        index=True,
    )

    user = relationship("User", back_populates="interactions")
    article = relationship("Article", back_populates="interactions")

    __table_args__ = (
        UniqueConstraint("user_id", "article_id", "interaction_type", name="uq_user_article_type"),
        Index("ix_interactions_user_time", user_id, created_at.desc()),
    )

    def __repr__(self) -> str:
        return f"<UserInteraction(user={self.user_id}, article={self.article_id}, type='{self.interaction_type}')>"
