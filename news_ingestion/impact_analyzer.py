"""
TechSage — Engineering impact analysis for AI releases.

Generates a structured engineering_impact JSONB for release types:
  model, framework, agent_platform, infra

Uses a two-layer approach:
1. Deterministic keyword rules for migration_risk and breaking_changes
   (these OVERRIDE any heuristic guess when triggered)
2. Heuristic classification via zero-shot (BART-MNLI from ModelRegistry)
   for drop_in_replacement and who_should_care

Output schema:
{
    "drop_in_replacement": "yes" | "no" | "partial" | "unknown",
    "breaking_changes": "none" | "minor" | "major" | "unknown",
    "migration_risk": "low" | "medium" | "high",
    "who_should_care": [list of engineer types],
    "action_required": short explanation
}
"""

import logging
import re

from config.model_registry import model_registry

logger = logging.getLogger(__name__)

_ELIGIBLE_RELEASE_TYPES = {"model", "framework", "agent_platform", "infra"}

# ── Deterministic keyword rules (highest priority) ──

_HIGH_RISK_PATTERNS = [
    (r"\bdeprecati(?:on|ng|ed)\b", "Deprecation detected"),
    (r"\bbreaking\s+change", "Breaking changes announced"),
    (r"\bpricing\s+increase", "Pricing increase"),
    (r"\bremoved?\s+(?:support|API|endpoint|feature)", "Feature/API removal"),
    (r"\bend[- ]?of[- ]?life\b", "End-of-life announced"),
    (r"\bsunset(?:ting|ted)?\b", "Service sunsetting"),
    (r"\bnot\s+backward[- ]?compatible\b", "Backward incompatibility"),
]

_MEDIUM_RISK_PATTERNS = [
    (r"\bnew\s+SDK\s+required\b", "New SDK required"),
    (r"\bmigrat(?:e|ion|ing)\s+(?:to|from|required|guide)", "Migration required"),
    (r"\bAPI\s+(?:v\d|version)\s+\d", "New API version"),
    (r"\brequires?\s+(?:update|upgrade)", "Update required"),
    (r"\bschema\s+change", "Schema change"),
    (r"\bnew\s+(?:authentication|auth)\s+(?:method|flow|required)", "Auth changes"),
    (r"\bformat\s+change", "Format change"),
]

_LOW_RISK_PATTERNS = [
    (r"\bbackward[- ]?compatible\b", "Backward compatible"),
    (r"\bdrop[- ]?in\s+replacement\b", "Drop-in replacement"),
    (r"\bno\s+breaking\s+change", "No breaking changes"),
    (r"\bfully\s+compatible\b", "Fully compatible"),
    (r"\bseamless\s+(?:upgrade|transition|migration)\b", "Seamless upgrade"),
    (r"\bopt(?:ional|[- ]?in)\b.*(?:upgrade|update|feature)", "Optional upgrade"),
]

# ── who_should_care mapping by release_type + keyword signals ──

_ROLE_SIGNALS: dict[str, list[tuple[str, str]]] = {
    "model": [
        (r"\bfine[- ]?tun", "ML Engineers"),
        (r"\bcontext|token|prompt", "Prompt Engineers"),
        (r"\bAPI|endpoint|SDK", "Backend Engineers"),
        (r"\bsafety|alignment|bias|guardrail", "AI Safety Engineers"),
        (r"\bbenchmark|eval|performance", "ML Engineers"),
        (r"\bcost|pricing|token.{0,10}price", "Engineering Managers"),
        (r"\bedge|mobile|on[- ]?device|quantiz", "Edge/Mobile Engineers"),
        (r"\bmultimodal|vision|image|audio|video", "Multimodal Engineers"),
    ],
    "framework": [
        (r"\bSDK|library|package|pip|npm", "Backend Engineers"),
        (r"\borchestrat|chain|workflow|pipeline", "ML Engineers"),
        (r"\bUI|frontend|component|widget", "Frontend Engineers"),
        (r"\bdeploy|infra|cloud|docker|k8s|kubernetes", "DevOps Engineers"),
        (r"\btest|debug|observ|monitor|trac", "Platform Engineers"),
        (r"\bagent|tool.?use|function.?call", "AI Engineers"),
    ],
    "agent_platform": [
        (r"\bagent|autonom|orchestrat", "AI Engineers"),
        (r"\btool|plugin|integration|connector", "Backend Engineers"),
        (r"\bsecurity|permission|auth|sandbox", "Security Engineers"),
        (r"\bdeploy|scale|infra", "DevOps Engineers"),
        (r"\bmemory|context|state|persist", "Backend Engineers"),
        (r"\bUI|chat|interface|dashboard", "Frontend Engineers"),
    ],
    "infra": [
        (r"\bGPU|TPU|compute|cluster|hardware", "Infrastructure Engineers"),
        (r"\bdeploy|serve|inference|latency", "MLOps Engineers"),
        (r"\bscale|autoscal|load|throughput", "Platform Engineers"),
        (r"\bcost|pricing|billing|spend", "Engineering Managers"),
        (r"\bsecurity|compliance|encrypt|audit", "Security Engineers"),
        (r"\bvector|embedding|search|index", "Data Engineers"),
        (r"\bmonitor|observ|log|metric|alert", "SRE Engineers"),
    ],
}

# Always-relevant roles per release type
_BASE_ROLES: dict[str, list[str]] = {
    "model": ["ML Engineers", "AI Engineers"],
    "framework": ["Backend Engineers", "AI Engineers"],
    "agent_platform": ["AI Engineers"],
    "infra": ["Infrastructure Engineers", "MLOps Engineers"],
}


def _score_migration_risk(text: str) -> tuple[str, str | None]:
    """
    Deterministic migration risk scoring.
    Returns (risk_level, reason) where reason is the first matched rule.
    """
    text_lower = text.lower()

    for pattern, reason in _HIGH_RISK_PATTERNS:
        if re.search(pattern, text_lower):
            return "high", reason

    for pattern, reason in _MEDIUM_RISK_PATTERNS:
        if re.search(pattern, text_lower):
            return "medium", reason

    for pattern, reason in _LOW_RISK_PATTERNS:
        if re.search(pattern, text_lower):
            return "low", reason

    return "low", None


def _score_breaking_changes(text: str) -> str:
    text_lower = text.lower()

    if any(re.search(p, text_lower) for p, _ in _HIGH_RISK_PATTERNS[:3]):
        return "major"

    if any(re.search(p, text_lower) for p, _ in _MEDIUM_RISK_PATTERNS):
        return "minor"

    if re.search(r"\bno\s+breaking", text_lower) or re.search(r"\bbackward[- ]?compatible", text_lower):
        return "none"

    return "unknown"


def _classify_drop_in(text: str) -> str:
    """Use zero-shot to classify drop-in replacement status."""
    if not model_registry.is_loaded:
        return "unknown"

    input_text = text[:512]
    labels = [
        "This is a drop-in replacement with no code changes needed",
        "This requires partial code changes to adopt",
        "This requires significant code changes or migration",
    ]

    try:
        result = model_registry.classifier_pipeline(
            input_text, candidate_labels=labels, multi_label=False
        )
        top = result["labels"][0]
        score = result["scores"][0]

        if score < 0.4:
            return "unknown"
        if "drop-in" in top:
            return "yes"
        if "partial" in top:
            return "partial"
        return "no"
    except Exception:
        return "unknown"


def _detect_who_should_care(text: str, release_type: str) -> list[str]:
    roles = set(_BASE_ROLES.get(release_type, []))
    text_lower = text.lower()

    signals = _ROLE_SIGNALS.get(release_type, [])
    for pattern, role in signals:
        if re.search(pattern, text_lower):
            roles.add(role)

    return sorted(roles)


def _generate_action_required(
    migration_risk: str,
    breaking_changes: str,
    risk_reason: str | None,
    release_type: str,
    title: str,
) -> str:
    if migration_risk == "high":
        reason_suffix = f": {risk_reason}" if risk_reason else ""
        return f"Immediate review required{reason_suffix}. Assess impact on current {release_type} integrations."

    if migration_risk == "medium":
        return f"Plan migration within current sprint. Review changelog for {release_type} compatibility."

    if breaking_changes == "none":
        return f"Low-risk update. Consider adopting when convenient for {release_type} improvements."

    return f"Monitor for relevance to your {release_type} stack. No immediate action needed."


def analyze_engineering_impact(
    title: str,
    cleaned_text: str,
    release_type: str,
) -> dict | None:
    """
    Generate structured engineering impact assessment.

    Args:
        title: Article title.
        cleaned_text: Full cleaned article body.
        release_type: Must be one of model/framework/agent_platform/infra.

    Returns:
        Engineering impact dict, or None if release_type is not eligible.
    """
    if release_type not in _ELIGIBLE_RELEASE_TYPES:
        return None

    full_text = f"{title}. {cleaned_text}"

    migration_risk, risk_reason = _score_migration_risk(full_text)
    breaking_changes = _score_breaking_changes(full_text)
    drop_in = _classify_drop_in(full_text)
    who_should_care = _detect_who_should_care(full_text, release_type)
    action_required = _generate_action_required(
        migration_risk, breaking_changes, risk_reason, release_type, title
    )

    # Deterministic overrides: if keyword rules fired, they take precedence
    if risk_reason and migration_risk == "high":
        if drop_in == "yes":
            drop_in = "partial"
        if breaking_changes == "none":
            breaking_changes = "major"
    elif risk_reason and migration_risk == "low":
        if drop_in == "no":
            drop_in = "partial"
        if breaking_changes in ("major", "minor"):
            breaking_changes = "none"

    impact = {
        "drop_in_replacement": drop_in,
        "breaking_changes": breaking_changes,
        "migration_risk": migration_risk,
        "who_should_care": who_should_care,
        "action_required": action_required,
    }

    logger.debug(
        "Engineering impact for '%s': risk=%s, breaking=%s, drop_in=%s, roles=%d",
        title[:60], migration_risk, breaking_changes, drop_in, len(who_should_care),
    )

    return impact
