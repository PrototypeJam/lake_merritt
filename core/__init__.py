"""
Core module for AI Evaluation Workbench.
Contains business logic for data processing, evaluation, and scoring.
"""

from core.data_models import (EvaluationItem, EvaluationMode,
                              EvaluationResults, ScorerResult)
from core.evaluation import run_evaluation
from core.generation import generate_outputs
# TODO: Update after ingestion refactor - from core.ingestion import load_evaluation_data
from core.reporting import results_to_csv, results_to_json

__all__ = [
    "EvaluationItem",
    "ScorerResult",
    "EvaluationResults",
    "EvaluationMode",
    "run_evaluation",
    "generate_outputs",
    # "load_evaluation_data",  # TODO: Add back after ingestion refactor
    "results_to_csv",
    "results_to_json",
]
