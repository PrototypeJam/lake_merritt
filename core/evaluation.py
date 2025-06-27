"""
Main evaluation orchestrator that coordinates the evaluation process.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Union

from core.data_models import (EvaluationItem, EvaluationResults, RunMetadata,
                              ScorerConfig)
from core.eval_pack import (EvalPackV1, PipelineExecutor, create_legacy_pack,
                            extract_scorer_configs, extract_selected_scorers)
from core.scoring import create_scorer, get_available_scorers

logger = logging.getLogger(__name__)


async def run_evaluation(
    items: List[EvaluationItem],
    selected_scorers: Optional[List[str]] = None,
    scorer_configs: Optional[Dict[str, Dict[str, Any]]] = None,
    api_keys: Optional[Dict[str, str]] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    pack: Optional[Union[EvalPackV1, Dict[str, Any]]] = None,
) -> EvaluationResults:
    """
    Run evaluation on a list of items using selected scorers.

    This function supports both legacy scorer-based evaluation and the new
    pack-based approach. If a pack is provided, it will be used directly.
    Otherwise, a legacy pack will be created from the provided scorers.

    Args:
        items: List of evaluation items
        selected_scorers: Names of scorers to use (required if pack not provided)
        scorer_configs: Configuration for each scorer (required if pack not provided)
        api_keys: API keys for LLM providers
        progress_callback: Optional callback for progress updates
        pack: Optional Eval Pack to use (can be EvalPackV1 or dict)

    Returns:
        EvaluationResults object containing all results and statistics
    """
    # Delegate to run_evaluation_batch with batch_size=1 for simplicity
    # This ensures consistent behavior between both functions
    return await run_evaluation_batch(
        items=items,
        selected_scorers=selected_scorers,
        scorer_configs=scorer_configs,
        api_keys=api_keys,
        batch_size=1,  # Process items one at a time
        progress_callback=progress_callback,
        pack=pack,
    )


async def run_evaluation_batch(
    items: List[EvaluationItem],
    selected_scorers: Optional[List[str]] = None,
    scorer_configs: Optional[Dict[str, Dict[str, Any]]] = None,
    api_keys: Optional[Dict[str, str]] = None,
    batch_size: int = 10,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    pack: Optional[Union[EvalPackV1, Dict[str, Any]]] = None,
) -> EvaluationResults:
    """
    Run evaluation in batches for better performance with async scorers.

    This function supports both legacy scorer-based evaluation and the new
    pack-based approach. If a pack is provided, it will be used directly.
    Otherwise, a legacy pack will be created from the provided scorers.

    Args:
        items: List of evaluation items
        selected_scorers: Names of scorers to use (required if pack not provided)
        scorer_configs: Configuration for each scorer (required if pack not provided)
        api_keys: API keys for LLM providers
        batch_size: Number of items to process in parallel
        progress_callback: Optional callback for progress updates
        pack: Optional Eval Pack to use (can be EvalPackV1 or dict)

    Returns:
        EvaluationResults object containing all results and statistics
    """
    start_time = datetime.now()
    
    # Handle pack parameter
    eval_pack = None
    if pack is not None:
        # If pack is a dict, convert it to EvalPackV1
        if isinstance(pack, dict):
            from core.eval_pack.loader import EvalPackLoader
            loader = EvalPackLoader()
            eval_pack, errors = loader.load(pack)
            if errors:
                logger.error(f"Errors loading pack: {errors}")
                raise ValueError(f"Invalid pack configuration: {errors}")
        else:
            eval_pack = pack
    else:
        # Create legacy pack from scorer selections
        if not selected_scorers:
            raise ValueError("Either 'pack' or 'selected_scorers' must be provided")
        
        api_keys = api_keys or {}
        scorer_configs = scorer_configs or {}
        
        eval_pack = create_legacy_pack(
            selected_scorers=selected_scorers,
            scorer_configs=scorer_configs,
            api_keys=api_keys,
            items=items,
        )
    
    logger.info(
        f"Starting pack-based evaluation with {len(items)} items, "
        f"batch size {batch_size}, pack: {eval_pack.name}"
    )
    
    # Create and initialize pipeline executor
    executor = PipelineExecutor(eval_pack)
    await executor.initialize(api_keys)
    
    # Run the pipeline
    processed_items = await executor.run_batch(
        items=items,
        batch_size=batch_size,
        progress_callback=progress_callback,
    )
    
    # Extract configuration for results
    if eval_pack.metadata.get("source") == "legacy_ui":
        # Use legacy format for backward compatibility
        selected_scorers = extract_selected_scorers(eval_pack)
        scorer_configs = extract_scorer_configs(eval_pack)
    else:
        # Use pack-based format
        selected_scorers = [stage.scorer for stage in eval_pack.pipeline]
        scorer_configs = {stage.scorer: stage.config for stage in eval_pack.pipeline}
    
    # Create results
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    results = EvaluationResults(
        items=processed_items,
        config={
            "scorers": selected_scorers,
            "scorer_configs": scorer_configs,
            "timestamp": start_time.isoformat(),
            "duration_seconds": duration,
            "batch_size": batch_size,
            "pack_name": eval_pack.name,
            "pack_version": eval_pack.version,
        },
        metadata={
            "mode": "evaluate_existing",
            "timestamp": start_time.isoformat(),
            "duration_seconds": duration,
            "total_items": len(items),
            "total_scorers": len(eval_pack.pipeline),
            "batch_size": batch_size,
            "pack_based": True,
        },
    )
    
    results.calculate_summary_stats()
    
    logger.info(f"Pack-based evaluation completed in {duration:.2f} seconds")
    return results
