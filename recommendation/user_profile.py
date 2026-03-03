"""
User profile retrieval and management.

Provides functions to access and interpret user preference profiles
for the ranking engine.
"""

import logging
from uuid import UUID

import numpy as np
from sqlalchemy.orm import Session

from config.settings import get_settings
from database.models import User, UserProfile

logger = logging.getLogger(__name__)


def get_or_create_user(db: Session, username: str) -> User:
    """Get an existing user by username, or create a new one."""
    user = db.query(User).filter(User.username == username).first()
    if not user:
        user = User(username=username)
        db.add(user)
        db.flush()
        logger.info("Created new user: %s (%s)", username, user.id)
    return user


def get_user_profile(db: Session, user_id: UUID) -> UserProfile | None:
    """Retrieve the user's preference profile."""
    return db.query(UserProfile).filter(UserProfile.user_id == user_id).first()


def get_category_vector(profile: UserProfile) -> dict[str, float]:
    """
    Return the user's normalized category preference vector.

    Returns a dict mapping each category to its weight [0, 1].
    Missing categories have weight 0.
    """
    settings = get_settings()
    all_categories = settings.categories

    if not profile or not profile.category_weights:
        # Uniform prior for new users — no category bias
        return {cat: 1.0 / len(all_categories) for cat in all_categories}

    weights = profile.category_weights
    return {cat: weights.get(cat, 0.0) for cat in all_categories}


def get_embedding_centroid(profile: UserProfile) -> np.ndarray | None:
    """
    Return the user's embedding centroid as a numpy array.

    Returns None if the user has no interaction history.
    """
    if not profile or profile.embedding_centroid is None:
        return None

    centroid = np.array(profile.embedding_centroid, dtype=np.float32)
    return centroid


def get_user_stats(db: Session, user_id: UUID) -> dict:
    """Return summary statistics for a user's profile."""
    profile = get_user_profile(db, user_id)

    if not profile:
        return {
            "interaction_count": 0,
            "category_preferences": {},
            "has_embedding_centroid": False,
        }

    return {
        "interaction_count": profile.interaction_count,
        "category_preferences": profile.category_weights or {},
        "has_embedding_centroid": profile.embedding_centroid is not None,
    }
