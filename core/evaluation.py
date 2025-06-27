"""
Main evaluation orchestrator that coordinates the evaluation process.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Union, IO
import pandas as pd

from core.data_models import (EvaluationItem, EvaluationResults, RunMetadata,
                              ScorerConfig, EvaluationMode)
from core.eval_pack import (EvalPackV1, PipelineExecutor, create_legacy_pack,
                            extract_scorer_configs, extract_selected_scorers)
from core.scoring import create_scorer, get_available_scorers
from core.ingestion import CSVIngester

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
    raw_data: Optional[Union[IO, pd.DataFrame]] = None,
    items: Optional[List[EvaluationItem]] = None,
    selected_scorers: Optional[List[str]] = None,
    scorer_configs: Optional[Dict[str, Dict[str, Any]]] = None,
    api_keys: Optional[Dict[str, str]] = None,
    batch_size: int = 10,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    pack: Optional[Union[EvalPackV1, Dict[str, Any]]] = None,
    mode: EvaluationMode = EvaluationMode.EVALUATE_EXISTING,
) -> EvaluationResults:
    """
    Run evaluation in batches for better performance with async scorers.
    
    This function supports both raw data ingestion and pre-processed items.
    If raw_data is provided, it will be ingested first using the appropriate ingester.
    
    Args:
        raw_data: Raw data (file object or DataFrame) to ingest
        items: List of evaluation items (used if raw_data is not provided)
        selected_scorers: Names of scorers to use (required if pack not provided)
        scorer_configs: Configuration for each scorer (required if pack not provided)
        api_keys: API keys for LLM providers
        batch_size: Number of items to process in parallel
        progress_callback: Optional callback for progress updates
        pack: Optional Eval Pack to use (can be EvalPackV1 or dict)
        mode: Evaluation mode (used for data ingestion)
    
    Returns:
        EvaluationResults object containing all results and statistics
    """
    # Handle raw data ingestion if provided
    if raw_data is not None:
        logger.info("Ingesting raw data before evaluation")
        ingester = CSVIngester()
        items = ingester.ingest(raw_data, {"mode": mode.value})
        logger.info(f"Ingested {len(items)} items from raw data")
    elif items is None:
        raise ValueError("Either 'raw_data' or 'items' must be provided")
    
    # Call the original implementation with the items
    return await _run_evaluation_batch_impl(
        items=items,
        selected_scorers=selected_scorers,
        scorer_configs=scorer_configs,
        api_keys=api_keys,
        batch_size=batch_size,
        progress_callback=progress_callback,
        pack=pack,
    )


async def _run_evaluation_batch_impl(
    items: List[EvaluationItem],
    selected_scorers: Optional[List[str]] = None,
    scorer_configs: Optional[Dict[str, Dict[str, Any]]] = None,
    api_keys: Optional[Dict[str, str]] = None,
    batch_size: int = 10,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    pack: Optional[Union[EvalPackV1, Dict[str, Any]]] = None,
) -> EvaluationResults:
    """
    Internal implementation of run_evaluation_batch.
    
    This is the original implementation that works with EvaluationItem objects.
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
    
    # Run the pipeline - it returns EvaluationResults
    results = await executor.run_batch(
        items=items,
        batch_size=batch_size,
        progress_callback=progress_callback,
    )
    
    logger.info(f"Pack-based evaluation completed")
    return results
