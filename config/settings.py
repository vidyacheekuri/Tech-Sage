"""
TechSage centralized configuration.
"""

from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    database_url: str = "postgresql://postgres:postgres@localhost:5432/techsage"

    # Optional API keys
    github_token: str = ""

    # ML Models
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    classifier_model: str = "facebook/bart-large-mnli"
    summarizer_model: str = "facebook/bart-large-cnn"
    embedding_dim: int = 384
    duplicate_threshold: float = 0.85

    # Ranking weights (must sum to 1.0)
    w_release_priority: float = 0.40
    w_embedding: float = 0.25
    w_recency: float = 0.20
    w_source_tier: float = 0.10
    w_category: float = 0.05

    # Topic categories for zero-shot classification
    categories: list[str] = [
        "LLMs & Foundation Models",
        "AI Agents & Assistants",
        "AI Infrastructure & MLOps",
        "AI Frameworks & Libraries",
        "AI Research & Papers",
        "AI Policy & Ethics",
        "AI Startups & Funding",
        "AI Applications",
    ]

    # Release type labels for secondary classification
    release_labels: list[str] = [
        "Model Release",
        "Framework or Library Release and Update",
        "Agent Platform",
        "Infrastructure Update",
        "Research Paper",
        "General AI News",
    ]

    summary_max_words: int = 60

    # Scheduler
    ingestion_interval_minutes: int = 5
    github_interval_hours: int = 6

    log_level: str = "INFO"

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


@lru_cache()
def get_settings() -> Settings:
    return Settings()
