"""
User interaction tracking and profile update.

When a user interacts with an article (view, click, bookmark, share),
this module:
1. Records the interaction event
2. Incrementally updates the user's preference profile

Incremental updates are key — we don't recompute from scratch on every interaction.
The centroid is updated using the running average formula:
    new_centroid = (old_centroid * n + new_embedding) / (n + 1)
This is O(1) per interaction instead of O(n).
"""

import logging
from uuid import UUID

import numpy as np
from sqlalchemy.orm import Session

from database.models import Article, User, UserInteraction, UserProfile

logger = logging.getLogger(__name__)

# Implicit weight of each interaction type for preference strength.
# Positive values boost the category/embedding; negative values dampen them.
INTERACTION_WEIGHTS = {
    "view": 1.0,
    "click": 2.0,
    "bookmark": 3.0,
    "share": 4.0,
    "like": 3.0,
    "dislike": -2.0,
}


def record_interaction(
    db: Session,
    user_id: UUID,
    article_id: UUID,
    interaction_type: str = "view",
) -> UserInteraction | None:
    """
    Record a user-article interaction and update the user's profile.

    Args:
        db: Database session.
        user_id: The user's ID.
        article_id: The article's ID.
        interaction_type: One of 'view', 'click', 'bookmark', 'share'.

    Returns:
        The created UserInteraction, or None if duplicate.
    """
    # Validate interaction type
    if interaction_type not in INTERACTION_WEIGHTS:
        logger.warning("Unknown interaction type: %s — defaulting to 'view'", interaction_type)
        interaction_type = "view"

    # Check for existing interaction of same type
    existing = (
        db.query(UserInteraction)
        .filter(
            UserInteraction.user_id == user_id,
            UserInteraction.article_id == article_id,
            UserInteraction.interaction_type == interaction_type,
        )
        .first()
    )
    if existing:
        logger.debug("Duplicate interaction skipped: user=%s, article=%s, type=%s", user_id, article_id, interaction_type)
        return None

    # Create interaction record
    interaction = UserInteraction(
        user_id=user_id,
        article_id=article_id,
        interaction_type=interaction_type,
    )
    db.add(interaction)

    # Fetch the article for profile update
    article = db.query(Article).filter(Article.id == article_id).first()
    if not article:
        logger.error("Article %s not found for interaction tracking.", article_id)
        db.flush()
        return interaction

    # Update user profile
    _update_user_profile(db, user_id, article, interaction_type)

    db.flush()
    return interaction


def _update_user_profile(
    db: Session,
    user_id: UUID,
    article: Article,
    interaction_type: str,
) -> None:
    """
    Incrementally update user profile with new interaction data.

    Updates:
    1. Category weights — weighted frequency histogram
    2. Embedding centroid — running average of interacted article embeddings
    3. Interaction count
    """
    profile = db.query(UserProfile).filter(UserProfile.user_id == user_id).first()

    if not profile:
        # Create new profile
        profile = UserProfile(user_id=user_id, category_weights={}, interaction_count=0)
        db.add(profile)

    weight = INTERACTION_WEIGHTS.get(interaction_type, 1.0)
    is_negative = weight < 0
    abs_weight = abs(weight)
    n = profile.interaction_count

    # --- Update category weights ---
    if article.category:
        cat_weights = dict(profile.category_weights) if profile.category_weights else {}

        if is_negative:
            # Dampen: reduce the disliked category's weight
            cat_weights[article.category] = max(0.0, cat_weights.get(article.category, 0.0) - abs_weight)
        else:
            # Boost: increase the liked category's weight
            cat_weights[article.category] = cat_weights.get(article.category, 0.0) + weight

        # Normalize to sum to 1.0 (only if there are positive weights)
        total = sum(cat_weights.values())
        if total > 0:
            cat_weights = {k: v / total for k, v in cat_weights.items()}

        profile.category_weights = cat_weights

    # --- Update embedding centroid ---
    if article.embedding is not None:
        article_emb = np.array(article.embedding, dtype=np.float32)

        if profile.embedding_centroid is not None and n > 0:
            old_centroid = np.array(profile.embedding_centroid, dtype=np.float32)

            if is_negative:
                # Push centroid AWAY from disliked article's embedding.
                # Subtract a fraction of the article embedding from the centroid.
                repulsion_strength = abs_weight / (n + abs_weight)
                new_centroid = old_centroid - repulsion_strength * article_emb
            else:
                # Pull centroid TOWARD liked article's embedding (running average).
                effective_n = n + weight
                new_centroid = (old_centroid * n + article_emb * weight) / effective_n
        else:
            if is_negative:
                # No centroid yet and first interaction is a dislike — skip embedding update
                new_centroid = None
            else:
                new_centroid = article_emb

        # L2 normalize the centroid for consistent cosine similarity
        if new_centroid is not None:
            norm = np.linalg.norm(new_centroid)
            if norm > 0:
                new_centroid = new_centroid / norm
            profile.embedding_centroid = new_centroid.tolist()

    profile.interaction_count = n + 1

    logger.debug(
        "Profile updated for user %s: %d interactions, categories=%s",
        user_id,
        profile.interaction_count,
        list((profile.category_weights or {}).keys()),
    )
