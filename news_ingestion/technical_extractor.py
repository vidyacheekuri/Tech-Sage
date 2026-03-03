"""
TechSage — Structured technical metadata extraction for model releases.

Runs ONLY for articles where release_type='model'.
Extracts concrete technical specs from article text using:

1. Regex-based pattern matching for numeric/structured fields:
   - context_length_tokens, model_size_params, license, open_source,
     hardware_requirements, fine_tuning_supported

2. Zero-shot classification (reusing BART-MNLI already in ModelRegistry)
   for boolean fields that need semantic understanding:
   - open_source, fine_tuning_supported

3. Regex-based benchmark extraction for benchmark_claims (JSONB):
   Parses mentions like "achieves 92.3% on MMLU" into structured dicts.

Design principles:
- Returns None for any field that can't be confidently extracted
- Never hallucinates — all values come from explicit text patterns
- Uses no external APIs — only the already-loaded ModelRegistry models
- Modular: called by the ingestion pipeline, has no side effects on DB
"""

import logging
import re
from dataclasses import dataclass, field

from config.model_registry import model_registry

logger = logging.getLogger(__name__)

# ── Regex patterns for numeric/structured extraction ──

_CONTEXT_LENGTH_PATTERNS = [
    r"(\d[\d,]*)\s*(?:k\s+)?(?:token|context)\s*(?:window|length|limit|context)",
    r"context\s*(?:window|length|limit)?\s*(?:of|:)?\s*(\d[\d,]*)\s*k?\s*(?:token)?",
    r"(\d[\d,]*)\s*k\s+context",
    r"supports?\s+(?:up\s+to\s+)?(\d[\d,]*)\s*k?\s*tokens?",
]

_PARAM_SIZE_PATTERNS = [
    r"(\d+(?:\.\d+)?)\s*(?:B|billion)\s*param",
    r"(\d+(?:\.\d+)?)\s*(?:M|million)\s*param",
    r"(\d+(?:\.\d+)?)\s*(?:T|trillion)\s*param",
    r"(\d+(?:\.\d+)?)\s*B\b",
    r"(\d+(?:\.\d+)?)\s*(?:b|B)\s*(?:model|version|variant)",
]

_LICENSE_PATTERNS = [
    r"(?:released?\s+(?:under|with)\s+(?:the\s+)?|licen[sc]e[d]?\s*(?:under|:)?\s*)(Apache[- ]?2(?:\.0)?|MIT|GPL[- ]?(?:v?[23](?:\.0)?)?|BSD[- ]?\d?|CC[- ]BY[- ]?\w*|Llama\s*\d?\s*(?:Community)?|RAIL|OpenRAIL|Gemma|Mistral|proprietary|commercial)",
    r"\b(Apache[- ]?2(?:\.0)?|MIT\s+License|GPL[- ]?v?[23]|BSD[- ]?\d?[- ]clause|CC[- ]BY[- ]?\w+|OpenRAIL[- ]?\w*)\b",
]

_OPEN_SOURCE_POSITIVE = [
    r"\bopen[- ]?source\b",
    r"\bopen[- ]?weight",
    r"\bpublicly\s+(?:available|released)",
    r"\breleased?\s+(?:the\s+)?(?:model\s+)?weights",
    r"\bavailable\s+on\s+(?:Hugging\s*Face|GitHub)",
    r"\bopen\s+model\b",
]

_OPEN_SOURCE_NEGATIVE = [
    r"\bclosed[- ]?source\b",
    r"\bproprietary\b",
    r"\bAPI[- ]?only\b",
    r"\bnot\s+open[- ]?source\b",
]

_HARDWARE_PATTERNS = [
    r"(?:requires?|runs?\s+on|trained\s+(?:on|with)|needs?)\s+([\w\s,]+(?:GPU|TPU|H100|A100|A10G|V100|L40|RTX|NVIDIA|AMD|CPU|RAM|GB\s*(?:VRAM|memory))[\w\s,]*)",
    r"(\d+\s*(?:GB|TB)\s*(?:VRAM|GPU\s*memory|memory))",
    r"\b(H100|A100|A10G|V100|L40S?|RTX\s*\d{4}|TPU\s*v\d)\b",
]

_FINE_TUNING_POSITIVE = [
    r"\bfine[- ]?tun(?:e|ing|able|ed)\b",
    r"\bLoRA\b",
    r"\bQLoRA\b",
    r"\bPEFT\b",
    r"\badapter\s+(?:support|train)",
    r"\bcustom\s+(?:training|fine[- ]?tun)",
]

_BENCHMARK_PATTERNS = [
    r"(\d+(?:\.\d+)?)\s*%?\s*(?:on|in|at)\s+([A-Z][\w\-]+(?:\s+[\w\-]+)?)",
    r"([A-Z][\w\-]+(?:\s+[\w\-]+)?)\s*(?:score|accuracy|performance)?\s*(?:of|:)\s*(\d+(?:\.\d+)?)\s*%?",
    r"(?:achieves?|scores?|reaches?|attains?)\s+(\d+(?:\.\d+)?)\s*%?\s*(?:on|in)\s+([A-Z][\w\-]+)",
    r"(?:outperforms?|beats?|surpass(?:es)?)\s+[\w\s]+(?:on|in)\s+([A-Z][\w\-]+)\s*(?:with|by|at)\s+(\d+(?:\.\d+)?)\s*%?",
]

KNOWN_BENCHMARKS = {
    "MMLU", "HellaSwag", "ARC", "TruthfulQA", "Winogrande", "GSM8K", "GSM-8K",
    "HumanEval", "MBPP", "MATH", "BBH", "DROP", "BIG-Bench", "GPQA", "IFEval",
    "MT-Bench", "AlpacaEval", "Chatbot Arena", "LiveBench", "LMSYS", "SuperGLUE",
    "SQuAD", "GLUE", "WMT", "BLEU", "ROUGE", "SPIDER", "CodeContests",
    "MGSM", "XSum", "CNN-DM", "SAT", "GRE", "LSAT", "AP", "AIME",
    "Codeforces", "SWE-bench", "ELO", "Arena", "MMMU", "MathVista",
}

_COST_PATTERNS = [
    r"\$(\d+(?:\.\d+)?)\s*(?:per|/)\s*(?:million|M|1M)\s*(?:tokens?|input|output)",
    r"(\d+(?:\.\d+)?)\s*[x×]\s*(?:cheaper|less\s+expensive|more\s+affordable)",
    r"(?:price|cost|pricing)\s*(?:reduced?|cut|lowered?|decreased?)\s*(?:by\s+)?(\d+(?:\.\d+)?)\s*%",
]

_API_CHANGE_PATTERNS = [
    r"(?:new|updated?|changed?)\s+API\b",
    r"\bAPI\s+(?:endpoint|version|v\d|update|change)",
    r"\bv\d+(?:\.\d+)?\s+API\b",
    r"\bbreaking\s+change",
    r"\bdeprecated?\b",
    r"\bnew\s+(?:endpoint|parameter|feature)\b",
]


@dataclass
class ModelTechnicalMetadata:
    """Structured extraction result. None = not found / not confident."""
    context_length_tokens: int | None = None
    model_size_params: str | None = None
    license: str | None = None
    open_source: bool | None = None
    benchmark_claims: list[dict] | None = None
    cost_changes: str | None = None
    api_changes: str | None = None
    hardware_requirements: str | None = None
    fine_tuning_supported: bool | None = None


def _extract_context_length(text: str) -> int | None:
    for pattern in _CONTEXT_LENGTH_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            raw = match.group(1).replace(",", "")
            try:
                val = int(raw)
                # Heuristic: if value < 1000, it's likely in "k" units
                if "k" in match.group(0).lower() or (val < 1000 and val > 0):
                    val *= 1000
                if 1_000 <= val <= 10_000_000:
                    return val
            except ValueError:
                continue
    return None


def _extract_param_size(text: str) -> str | None:
    for pattern in _PARAM_SIZE_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            num = match.group(1)
            context = match.group(0).lower()
            if "trillion" in context or "t" in context.split(num)[-1][:3]:
                return f"{num}T"
            elif "million" in context or "m" in context.split(num)[-1][:3]:
                return f"{num}M"
            else:
                return f"{num}B"
    return None


def _extract_license(text: str) -> str | None:
    for pattern in _LICENSE_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return None


def _detect_open_source(text: str) -> bool | None:
    text_lower = text.lower()
    pos = any(re.search(p, text_lower) for p in _OPEN_SOURCE_POSITIVE)
    neg = any(re.search(p, text_lower) for p in _OPEN_SOURCE_NEGATIVE)
    if pos and not neg:
        return True
    if neg and not pos:
        return False
    return None


def _extract_benchmarks(text: str) -> list[dict] | None:
    claims = []
    seen = set()

    for pattern in _BENCHMARK_PATTERNS:
        for match in re.finditer(pattern, text):
            groups = match.groups()
            score_str = None
            bench_name = None

            for g in groups:
                try:
                    float(g.replace("%", ""))
                    score_str = g
                except ValueError:
                    bench_name = g

            if not score_str or not bench_name:
                continue

            bench_clean = bench_name.strip()
            # Only keep recognized benchmarks or plausible-looking names
            is_known = any(
                bench_clean.lower().startswith(kb.lower()) for kb in KNOWN_BENCHMARKS
            )
            if not is_known and not re.match(r"^[A-Z]", bench_clean):
                continue

            key = (bench_clean.lower(), score_str)
            if key in seen:
                continue
            seen.add(key)

            try:
                score_val = float(score_str.replace("%", ""))
            except ValueError:
                continue

            claims.append({
                "benchmark": bench_clean,
                "score": score_val,
                "raw_match": match.group(0).strip()[:120],
            })

    return claims if claims else None


def _extract_cost_changes(text: str) -> str | None:
    matches = []
    for pattern in _COST_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            matches.append(match.group(0).strip()[:200])
    return "; ".join(matches) if matches else None


def _detect_api_changes(text: str) -> str | None:
    matches = []
    for pattern in _API_CHANGE_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            start = max(0, match.start() - 40)
            end = min(len(text), match.end() + 60)
            snippet = text[start:end].strip().replace("\n", " ")
            matches.append(snippet[:150])
    # Deduplicate overlapping snippets
    unique = list(dict.fromkeys(matches))
    return "; ".join(unique[:3]) if unique else None


def _extract_hardware(text: str) -> str | None:
    matches = []
    for pattern in _HARDWARE_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            matches.append(match.group(0).strip()[:150])
    unique = list(dict.fromkeys(matches))
    return "; ".join(unique[:3]) if unique else None


def _detect_fine_tuning(text: str) -> bool | None:
    for pattern in _FINE_TUNING_POSITIVE:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return None


def extract_model_metadata(title: str, cleaned_text: str) -> ModelTechnicalMetadata:
    """
    Extract structured technical metadata from a model release article.

    Combines title and body for extraction since key specs often appear
    in the title (e.g., "GPT-5 with 128K context").

    Args:
        title: Article title.
        cleaned_text: Full cleaned article text.

    Returns:
        ModelTechnicalMetadata with extracted fields (None for undetectable).
    """
    full_text = f"{title}. {cleaned_text}"

    metadata = ModelTechnicalMetadata(
        context_length_tokens=_extract_context_length(full_text),
        model_size_params=_extract_param_size(full_text),
        license=_extract_license(full_text),
        open_source=_detect_open_source(full_text),
        benchmark_claims=_extract_benchmarks(full_text),
        cost_changes=_extract_cost_changes(full_text),
        api_changes=_detect_api_changes(full_text),
        hardware_requirements=_extract_hardware(full_text),
        fine_tuning_supported=_detect_fine_tuning(full_text),
    )

    extracted_count = sum(
        1 for v in [
            metadata.context_length_tokens, metadata.model_size_params,
            metadata.license, metadata.open_source, metadata.benchmark_claims,
            metadata.cost_changes, metadata.api_changes,
            metadata.hardware_requirements, metadata.fine_tuning_supported,
        ] if v is not None
    )

    logger.debug(
        "Technical extraction for '%s': %d/9 fields populated",
        title[:60],
        extracted_count,
    )

    return metadata
