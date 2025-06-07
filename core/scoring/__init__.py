"""Scoring modules."""
from importlib import import_module
from typing import Any, Callable, Dict

__all__ = [
    "BaseScorer",
    "create_scorer",
    "get_available_scorers",
    "exact_match",
    "fuzzy_match",
    "llm_judge",
]


class BaseScorer:
    """Base class for all scorers."""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}

    def score(self, item):  # pragma: no cover - to be overridden
        raise NotImplementedError


class _FunctionScorer(BaseScorer):
    """Wrap a simple function-based scorer in a class."""

    def __init__(self, func: Callable, config: Dict[str, Any] = None):
        super().__init__(config)
        self.func = func

    def score(self, item):
        return self.func(item, self.config)


def get_available_scorers() -> Dict[str, Callable]:
    return {
        name: import_module(f"core.scoring.{name}")
        for name in ["exact_match", "fuzzy_match", "llm_judge"]
    }


def create_scorer(name: str, config: Dict[str, Any] | None = None) -> BaseScorer:
    module = import_module(f"core.scoring.{name}")
    if hasattr(module, "Scorer"):
        return module.Scorer(config or {})
    elif hasattr(module, "LLMJudgeScorer"):
        return module.LLMJudgeScorer(config or {})
    elif hasattr(module, "score"):
        return _FunctionScorer(module.score, config or {})
    else:
        raise ValueError(f"Unsupported scorer module: {name}")
