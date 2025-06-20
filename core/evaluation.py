"""
Main evaluation orchestrator that coordinates the evaluation process.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from core.data_models import (EvaluationItem, EvaluationResults, RunMetadata,
                              ScorerConfig)
from core.scoring import create_scorer, get_available_scorers

logger = logging.getLogger(__name__)


async def run_evaluation(
    items: List[EvaluationItem],
    selected_scorers: List[str],
    scorer_configs: Dict[str, Dict[str, Any]],
    api_keys: Dict[str, str],
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> EvaluationResults:
    """
    Run evaluation on a list of items using selected scorers.

    Args:
        items: List of evaluation items
        selected_scorers: Names of scorers to use
        scorer_configs: Configuration for each scorer
        api_keys: API keys for LLM providers
        progress_callback: Optional callback for progress updates

    Returns:
        EvaluationResults object containing all results and statistics
    """
    start_time = datetime.now()
    logger.info(
        f"Starting evaluation with {len(items)} items and {len(selected_scorers)} scorers"
    )

    # Create scorer instances
    scorers = {}
    for scorer_name in selected_scorers:
        config = scorer_configs.get(scorer_name, {})
        # Add API keys to config if needed
        if scorer_name == "llm_judge" and "api_key" not in config:
            provider = config.get("provider", "openai")
            config["api_key"] = api_keys.get(provider)

        try:
            scorer = create_scorer(scorer_name, config)
            scorers[scorer_name] = scorer
            logger.info(f"Created scorer: {scorer_name}")
        except Exception as e:
            logger.error(f"Failed to create scorer {scorer_name}: {e}")
            raise

    # Process each item
    total_operations = len(items) * len(scorers)
    completed_operations = 0

    for item_idx, item in enumerate(items):
        # Clear existing scores
        item.scores = []

        # Apply each scorer
        for scorer_name, scorer in scorers.items():
            try:
                # Run scorer
                if asyncio.iscoroutinefunction(scorer.score):
                    result = await scorer.score(item)
                else:
                    result = scorer.score(item)

                item.scores.append(result)
                logger.debug(
                    f"Scored item {item_idx} with {scorer_name}: {result.score}"
                )

            except Exception as e:
                logger.error(f"Error scoring item {item_idx} with {scorer_name}: {e}")
                # Add error result
                from core.data_models import ScorerResult

                item.scores.append(
                    ScorerResult(
                        scorer_name=scorer_name,
                        score=0.0,
                        passed=False,
                        error=str(e),
                    )
                )

            # Update progress
            completed_operations += 1
            if progress_callback:
                progress_callback(completed_operations, total_operations)

    # Create results object
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    results = EvaluationResults(
        items=items,
        config={
            "scorers": selected_scorers,
            "scorer_configs": scorer_configs,
            "timestamp": start_time.isoformat(),
            "duration_seconds": duration,
        },
        metadata={
            "mode": "evaluate_existing",  # Will be updated by caller if different
            "timestamp": start_time.isoformat(),
            "duration_seconds": duration,
            "total_items": len(items),
            "total_scorers": len(scorers),
        },
    )

    # Calculate summary statistics
    results.calculate_summary_stats()

    logger.info(f"Evaluation completed in {duration:.2f} seconds")
    return results


async def run_evaluation_batch(
    items: List[EvaluationItem],
    selected_scorers: List[str],
    scorer_configs: Dict[str, Dict[str, Any]],
    api_keys: Dict[str, str],
    batch_size: int = 10,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> EvaluationResults:
    """
    Run evaluation in batches for better performance with async scorers.

    This is an optimized version that processes items in batches,
    particularly useful for LLM-based scorers.
    """
    start_time = datetime.now()
    logger.info(
        f"Starting batch evaluation with {len(items)} items, batch size {batch_size}"
    )

    # Create scorer instances
    scorers = {}
    for scorer_name in selected_scorers:
        config = scorer_configs.get(scorer_name, {})
        # Add API key for LLM-based scorers
        if (
            scorer_name in ["llm_judge", "criteria_selection_judge"]
            and "api_key" not in config
        ):
            provider = config.get("provider", "openai")
            config["api_key"] = api_keys.get(provider)

        scorers[scorer_name] = create_scorer(scorer_name, config)

    # Process in batches
    total_operations = len(items) * len(scorers)
    completed_operations = 0

    for i in range(0, len(items), batch_size):
        batch = items[i : i + batch_size]

        # Create tasks for this batch
        tasks = []
        for item in batch:
            item.scores = []
            for scorer_name, scorer in scorers.items():
                if asyncio.iscoroutinefunction(scorer.score):
                    task = scorer.score(item)
                else:
                    # Wrap sync function in coroutine
                    task = asyncio.create_task(asyncio.to_thread(scorer.score, item))
                tasks.append((item, scorer_name, task))

        # Wait for batch to complete
        for item, scorer_name, task in tasks:
            try:
                result = await task
                item.scores.append(result)
            except Exception as e:
                logger.error(f"Error in batch scoring: {e}")
                from core.data_models import ScorerResult

                item.scores.append(
                    ScorerResult(
                        scorer_name=scorer_name,
                        score=0.0,
                        passed=False,
                        error=str(e),
                    )
                )

            completed_operations += 1
            if progress_callback:
                progress_callback(completed_operations, total_operations)

    # Create results
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    results = EvaluationResults(
        items=items,
        config={
            "scorers": selected_scorers,
            "scorer_configs": scorer_configs,
            "timestamp": start_time.isoformat(),
            "duration_seconds": duration,
            "batch_size": batch_size,
        },
        metadata={
            "mode": "evaluate_existing",
            "timestamp": start_time.isoformat(),
            "duration_seconds": duration,
            "total_items": len(items),
            "total_scorers": len(scorers),
            "batch_size": batch_size,
        },
    )

    results.calculate_summary_stats()

    logger.info(f"Batch evaluation completed in {duration:.2f} seconds")
    return results
