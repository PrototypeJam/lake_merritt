"""
Reporting utilities for converting evaluation results to various formats.
"""
import json
import pandas as pd
import csv
import io
from typing import Dict, Any, List, Set
from datetime import datetime
import numpy as np

from core.data_models import EvaluationResults, EvaluationItem


def results_to_csv(results: EvaluationResults) -> str:
    """
    Convert evaluation results to CSV format.
    
    Args:
        results: Evaluation results object
    
    Returns:
        CSV string
    """
    if not results.items:
        return ""
    
    # Collect all metadata keys and scorer names
    metadata_keys: Set[str] = set()
    scorer_names: Set[str] = set()
    
    for item in results.items:
        metadata_keys.update(item.metadata.keys())
        for score in item.scores:
            scorer_names.add(score.scorer_name)
    
    # Build fieldnames
    fieldnames = ["id", "input", "output", "expected_output"]
    
    # Add metadata columns
    for key in sorted(metadata_keys):
        fieldnames.append(f"metadata_{key}")
    
    # Add scorer fields
    for scorer in sorted(scorer_names):
        fieldnames.extend([
            f"{scorer}_score",
            f"{scorer}_passed",
            f"{scorer}_reasoning",
            f"{scorer}_error"
        ])
    
    # Write CSV
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    
    for item in results.items:
        row = {
            "id": item.id or "",
            "input": item.input,
            "output": item.output or "",
            "expected_output": item.expected_output,
        }
        
        # Add metadata
        for key in metadata_keys:
            row[f"metadata_{key}"] = item.metadata.get(key, "")
        
        # Add scores
        for score in item.scores:
            row[f"{score.scorer_name}_score"] = score.score
            row[f"{score.scorer_name}_passed"] = score.passed
            row[f"{score.scorer_name}_reasoning"] = score.reasoning or ""
            if score.error:
                row[f"{score.scorer_name}_error"] = score.error
        
        writer.writerow(row)
    
    return output.getvalue()


def results_to_json(results: EvaluationResults) -> str:
    """
    Convert evaluation results to JSON format.
    
    Args:
        results: Evaluation results object
    
    Returns:
        JSON string
    """
    return results.model_dump_json(indent=2)


def generate_summary_report(results: EvaluationResults) -> str:
    """
    Generate a human-readable summary report in Markdown format.
    
    Args:
        results: Evaluation results object
    
    Returns:
        Markdown-formatted report string
    """
    if not results.items:
        return "# Evaluation Summary Report\n\nNo evaluation results to summarize."

    report_lines = []
    
    # Header
    report_lines.append("# Evaluation Summary Report")
    report_lines.append("")
    report_lines.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    # Overview
    report_lines.append("## Overview")
    report_lines.append("")
    report_lines.append(f"- **Total Items Evaluated**: {len(results.items)}")
    report_lines.append(f"- **Evaluation Mode**: {results.metadata.get('mode', 'Unknown')}")
    report_lines.append(f"- **Duration**: {results.metadata.get('duration_seconds', 0):.2f} seconds")
    report_lines.append(f"- **Scorers Used**: {', '.join(results.summary_stats.keys())}")
    report_lines.append("")
    
    # Summary Statistics
    report_lines.append("## Summary Statistics")
    report_lines.append("")
    
    for scorer_name, stats in results.summary_stats.items():
        report_lines.append(f"### {scorer_name.replace('_', ' ').title()}")
        report_lines.append("")
        report_lines.append(f"- **Accuracy**: {stats.get('accuracy', 0):.1%}")
        report_lines.append(f"- **Items Passed**: {stats.get('passed', 0)}/{stats.get('total', 0)}")
        report_lines.append(f"- **Items Failed**: {stats.get('failed', 0)}/{stats.get('total', 0)}")
        
        if stats.get('errors', 0) > 0:
            report_lines.append(f"- **Errors**: {stats.get('errors', 0)}")
        
        report_lines.append(f"- **Average Score**: {stats.get('average_score', 0):.3f}")
        report_lines.append(f"- **Score Range**: {stats.get('min_score', 0):.3f} - {stats.get('max_score', 0):.3f}")
        
        # Score distribution if available
        if 'score_distribution' in stats:
            report_lines.append("")
            report_lines.append("**Score Distribution:**")
            for range_label, count in stats['score_distribution'].items():
                report_lines.append(f"  - {range_label}: {count} items")
        
        report_lines.append("")
    
    # Failure Analysis
    report_lines.append("## Failure Analysis")
    report_lines.append("")
    
    # Collect failures by scorer
    failures_by_scorer = {}
    for item in results.items:
        for score in item.scores:
            if not score.passed and not score.error:
                if score.scorer_name not in failures_by_scorer:
                    failures_by_scorer[score.scorer_name] = []
                failures_by_scorer[score.scorer_name].append({
                    "id": item.id,
                    "score": score.score,
                    "reasoning": score.reasoning,
                })
    
    if failures_by_scorer:
        for scorer_name, failures in failures_by_scorer.items():
            report_lines.append(f"### {scorer_name.replace('_', ' ').title()} Failures")
            report_lines.append("")
            report_lines.append(f"Total failures: {len(failures)}")
            report_lines.append("")
            
            # Show top 5 failures
            for failure in failures[:5]:
                report_lines.append(f"- **Item {failure['id']}** (Score: {failure['score']:.3f})")
                if failure['reasoning']:
                    report_lines.append(f"  - Reason: {failure['reasoning'][:100]}...")
            
            if len(failures) > 5:
                report_lines.append(f"- ... and {len(failures) - 5} more failures")
            
            report_lines.append("")
    else:
        report_lines.append("No failures detected across all scorers.")
        report_lines.append("")
    
    # Configuration Used
    report_lines.append("## Configuration")
    report_lines.append("")
    report_lines.append("```json")
    report_lines.append(json.dumps(results.config, indent=2))
    report_lines.append("```")
    report_lines.append("")
    
    # Recommendations
    report_lines.append("## Recommendations")
    report_lines.append("")
    
    # Generate recommendations based on results
    recommendations = generate_recommendations(results)
    for rec in recommendations:
        report_lines.append(f"- {rec}")
    
    return "\n".join(report_lines)


def generate_recommendations(results: EvaluationResults) -> List[str]:
    """
    Generate recommendations based on evaluation results.
    
    Args:
        results: Evaluation results object
    
    Returns:
        List of recommendation strings
    """
    recommendations = []
    
    # Check overall performance
    avg_accuracy = sum(
        stats.get('accuracy', 0) for stats in results.summary_stats.values()
    ) / len(results.summary_stats) if results.summary_stats else 0
    
    if avg_accuracy < 0.5:
        recommendations.append(
            "Overall accuracy is below 50%. Consider reviewing the model's training data or prompts."
        )
    elif avg_accuracy < 0.8:
        recommendations.append(
            "There's room for improvement. Focus on the specific failure cases to identify patterns."
        )
    else:
        recommendations.append(
            "Good overall performance! Consider adding more challenging test cases."
        )
    
    # Check for scorer-specific issues
    for scorer_name, stats in results.summary_stats.items():
        if stats.get('errors', 0) > 0:
            recommendations.append(
                f"The {scorer_name} scorer encountered errors. Check API limits or configuration."
            )
        
        if scorer_name == "exact_match" and stats.get('accuracy', 0) < 0.3:
            recommendations.append(
                "Low exact match scores. Consider using fuzzy matching for more flexibility."
            )
        
        if scorer_name == "llm_judge" and stats.get('average_score', 0) < 0.5:
            recommendations.append(
                "LLM judge scores are low. Review the judge prompt for clarity and criteria."
            )
    
    # Check for consistency across scorers
    if len(results.summary_stats) > 1:
        accuracies = [stats.get('accuracy', 0) for stats in results.summary_stats.values()]
        variance = max(accuracies) - min(accuracies)
        if variance > 0.3:
            recommendations.append(
                "Large variance in scorer results. Ensure all scorers are properly configured and aligned."
            )
    
    return recommendations


def export_detailed_analysis(
    results: EvaluationResults,
    output_path: str,
    include_all_items: bool = False,
) -> None:
    """
    Export a detailed analysis to a file.
    
    Args:
        results: Evaluation results object
        output_path: Path to save the analysis
        include_all_items: Whether to include all items or just failures
    """
    if not results.items:
        raise ValueError("No evaluation items to export.")

    with open(output_path, 'w') as f:
        # Write summary
        f.write(generate_summary_report(results))
        f.write("\n\n")
        
        # Write detailed item analysis
        f.write("# Detailed Item Analysis\n\n")
        
        items_to_analyze = results.items
        if not include_all_items:
            # Filter to items with at least one failure
            items_to_analyze = [
                item for item in results.items
                if any(not score.passed for score in item.scores)
            ]
        
        for item in items_to_analyze:
            f.write(f"## Item: {item.id}\n\n")
            f.write(f"**Input:**\n```\n{item.input}\n```\n\n")
            f.write(f"**Expected Output:**\n```\n{item.expected_output}\n```\n\n")
            f.write(f"**Actual Output:**\n```\n{item.output}\n```\n\n")
            
            f.write("**Scores:**\n")
            for score in item.scores:
                status = "✅ PASS" if score.passed else "❌ FAIL"
                f.write(f"- {score.scorer_name}: {status} (Score: {score.score:.3f})\n")
                if score.reasoning:
                    f.write(f"  - Reasoning: {score.reasoning}\n")
                if score.error:
                    f.write(f"  - Error: {score.error}\n")
            
            f.write("\n---\n\n")
