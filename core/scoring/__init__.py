"""Scoring utilities and factory functions."""
from __future__ import annotations

from typing import Any, Dict, Type

from .exact_match import ExactMatchScorer, CaseInsensitiveExactMatchScorer, NormalizedExactMatchScorer
from .fuzzy_match import FuzzyMatchScorer
from .llm_judge import LLMJudgeScorer


_SCORERS: Dict[str, Dict[str, Any]] = {
    "exact_match": {
        "class": ExactMatchScorer,
        "display_name": "Exact Match",
        "description": "Checks for strict equality between output and expected output.",
    },
    "case_insensitive_exact_match": {
        "class": CaseInsensitiveExactMatchScorer,
        "display_name": "Case-Insensitive Exact Match",
        "description": "Exact match ignoring case differences.",
    },
    "normalized_exact_match": {
        "class": NormalizedExactMatchScorer,
        "display_name": "Normalized Exact Match",
        "description": "Exact match after text normalization.",
    },
    "fuzzy_match": {
        "class": FuzzyMatchScorer,
        "display_name": "Fuzzy Match",
        "description": "Computes similarity ratio between texts.",
    },
    "llm_judge": {
        "class": LLMJudgeScorer,
        "display_name": "LLM Judge",
        "description": "Uses a language model to judge correctness.",
    },
}


def get_available_scorers() -> Dict[str, Dict[str, Any]]:
    """Return metadata about available scorers."""
    return _SCORERS


def create_scorer(name: str, config: Optional[Dict[str, Any]] = None):
    """Instantiate a scorer by name."""
    if name not in _SCORERS:
        raise ValueError(f"Unknown scorer: {name}")
    cls: Type[Any] = _SCORERS[name]["class"]
    return cls(config)
