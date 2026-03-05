"""
TechSage — RAG query pipeline.

Flow: embed query → vector search → build context → generate answer with local LLM.
"""

import logging
from typing import Any

from sqlalchemy.orm import Session

from config.model_registry import model_registry
from database.models import Article
from news_ingestion.embedding import generate_embedding

logger = logging.getLogger(__name__)

# Number of articles to retrieve for context
RAG_TOP_K = 6

# Max tokens for LLM response (kept low for fast CPU inference)
RAG_MAX_NEW_TOKENS = 100


def _ensure_rag_model_loaded() -> None:
    """Load RAG LLM on first use if not already loaded."""
    if not model_registry.rag_model_loaded:
        model_registry.load_rag_model()


def _retrieve_articles(db: Session, query_embedding: list[float], top_k: int = RAG_TOP_K) -> list[Article]:
    """Retrieve top-k articles by embedding similarity, excluding duplicates."""
    from sqlalchemy import select

    dist = Article.embedding.cosine_distance(query_embedding)
    stmt = (
        select(Article)
        .where(Article.embedding.isnot(None), Article.duplicate_of.is_(None))
        .order_by(dist)
        .limit(top_k)
    )
    result = db.execute(stmt)
    return list(result.scalars().all())


def _build_context(articles: list[Article]) -> str:
    """Build context string from retrieved articles for the LLM (trimmed for speed)."""
    parts = []
    for i, a in enumerate(articles, 1):
        summary = (a.summary or a.title or "")[:250]
        parts.append(f"[{i}] {a.title} | {a.source} | {summary}")
    return "\n".join(parts) if parts else "No relevant articles found."


def _generate_answer(question: str, context: str) -> str:
    """Generate answer using local LLM with retrieved context."""
    _ensure_rag_model_loaded()
    tokenizer = model_registry.rag_tokenizer
    model = model_registry.rag_model

    # SmolLM2 chat format: <|im_start|>system\n...<|im_end|>\n<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n
    prompt = f"""<|im_start|>system
You are TechSage. Answer based ONLY on the articles below. Be concise (2-3 sentences). Cite [1], [2].<|im_end|>
<|im_start|>user
Articles:
{context}

Question: {question}<|im_end|>
<|im_start|>assistant
"""
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1536,
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=RAG_MAX_NEW_TOKENS,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

    reply = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    return reply.strip()


def rag_query(db: Session, question: str) -> dict[str, Any]:
    """
    Run RAG pipeline: embed → retrieve → generate.

    Returns:
        {"answer": str, "sources": [{"title", "url", "source"}, ...]}
    """
    if not question or not question.strip():
        return {"answer": "Please ask a question about AI releases, frameworks, or research.", "sources": []}

    question = question.strip()

    if not model_registry.is_loaded:
        raise RuntimeError("Base models not loaded. Ensure model_registry.load_all() ran at startup.")

    # 1. Embed query
    query_emb = generate_embedding(question)
    query_list = query_emb.tolist()

    # 2. Retrieve
    articles = _retrieve_articles(db, query_list)
    if not articles:
        return {
            "answer": "No relevant articles found in the TechSage corpus. Try a different question or wait for more articles to be ingested.",
            "sources": [],
        }

    # 3. Build context
    context = _build_context(articles)

    # 4. Generate
    try:
        answer = _generate_answer(question, context)
    except Exception as e:
        logger.exception("RAG generation failed: %s", e)
        answer = f"Sorry, I couldn't generate an answer. Error: {e}"

    # 5. Format sources
    sources = [{"title": a.title, "url": a.url, "source": a.source} for a in articles]

    return {"answer": answer, "sources": sources}
