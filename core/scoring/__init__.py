"""Scoring modules."""
from importlib import import_module
from typing import Dict, Callable

__all__ = ["exact_match", "fuzzy_match", "llm_judge", "list_scorers"]


def list_scorers() -> Dict[str, Callable]:
    return {name: import_module(f"core.scoring.{name}") for name in ["exact_match", "fuzzy_match", "llm_judge"]}
