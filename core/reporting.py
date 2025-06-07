"""Reporting utilities for evaluation results."""
from __future__ import annotations

from typing import Dict, Any
import csv
import io
import json

from .data_models import EvaluationResults
from .scoring import get_available_scorers


def results_to_csv(results: EvaluationResults) -> str:
    """Convert evaluation results to CSV string."""
    metadata_keys = set()
    scorer_names = []
    for item in results.items:
        metadata_keys.update(item.metadata.keys())
        scorer_names.extend([score.scorer_name for score in item.scores])
    scorer_names = list(dict.fromkeys(scorer_names))

    fieldnames = ["id", "input", "output", "expected_output"]
    for key in sorted(metadata_keys):
        fieldnames.append(f"metadata_{key}")
    for name in scorer_names:
        fieldnames.append(f"{name}_score")
        fieldnames.append(f"{name}_passed")

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    for item in results.items:
        row: Dict[str, Any] = {
            "id": item.id,
            "input": item.input,
            "output": item.output,
            "expected_output": item.expected_output,
        }
        for key in metadata_keys:
            row[f"metadata_{key}"] = item.metadata.get(key)
        for score in item.scores:
            row[f"{score.scorer_name}_score"] = score.score
            row[f"{score.scorer_name}_passed"] = score.passed
        writer.writerow(row)
    return output.getvalue()


def results_to_json(results: EvaluationResults) -> str:
    """Serialize results to JSON string."""
    return results.model_dump_json(indent=2)


def generate_summary_report(results: EvaluationResults) -> str:
    """Generate a simple Markdown summary report."""
    lines = ["# Evaluation Summary Report", f"Total Items Evaluated: {len(results.items)}"]
    scorer_info = get_available_scorers()
    for name, stats in results.summary_stats.items():
        display = scorer_info.get(name, {}).get("display_name", name.replace("_", " ").title())
        accuracy = stats.get("accuracy", 0) * 100
        lines.append(f"## {display}")
        lines.append(f"Items Passed: {stats['passed']}/{stats['total']}")
        lines.append(f"Accuracy: {accuracy:.1f}%")
    return "\n".join(lines)
