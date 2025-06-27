"""
Core module for AI Evaluation Workbench.
Contains business logic for data processing, evaluation, and scoring.
"""

from core.data_models import (EvaluationItem, EvaluationMode,
                              EvaluationResults, ScorerResult)
from core.evaluation import run_evaluation
from core.generation import generate_outputs
from core.reporting import results_to_csv, results_to_json
from core.registry import ComponentRegistry

# Bootstrap the component registry at module load time
ComponentRegistry.discover_builtins()

__all__ = [
    "EvaluationItem",
    "ScorerResult",
    "EvaluationResults",
    "EvaluationMode",
    "run_evaluation",
    "generate_outputs",
    "results_to_csv",
    "results_to_json",
]
