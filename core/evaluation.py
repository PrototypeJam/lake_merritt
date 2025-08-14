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
# FIX: Import all necessary components for the new workflow
from core.eval_pack import (EvalPackV1, PipelineExecutor, create_legacy_pack,
                            extract_scorer_configs, extract_selected_scorers)
from core.scoring import create_scorer, get_available_scorers
from core.ingestion import CSVIngester
from core.registry import ComponentRegistry
from core.eval_pack.schema import GenerationMode # <-- Import GenerationMode
from core.generation_handler import handle_mode_b_generation, handle_mode_b_generation_from_items # <-- Import both handlers


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
    user_context: Optional[str] = None, # <-- Add user_context parameter
) -> EvaluationResults:
    """
    Ingest data (if needed) and run a pack-based evaluation, preserving the
    legacy scorer path when no pack is supplied.
    """

    # ------------------------------------------------------------------
    # 1. Resolve which evaluation pack we are using **before** ingestion
    # ------------------------------------------------------------------
    eval_pack: EvalPackV1
    if pack is not None:
        if isinstance(pack, dict):
            from core.eval_pack.loader import EvalPackLoader
            loader = EvalPackLoader()
            eval_pack, errors = loader.load(pack)
            if errors:
                logger.error(f"Pack validation errors: {errors}")
                raise ValueError(f"Invalid pack configuration: {errors}")
        else:
            eval_pack = pack
    else:
        # Fallback to legacy one-off pack built from UI scorer choices
        if not selected_scorers:
            raise ValueError("Either 'pack' or 'selected_scorers' must be provided")
        
        eval_pack = create_legacy_pack(
            selected_scorers=selected_scorers,
            scorer_configs=scorer_configs or {},
            api_keys=api_keys or {},
            items=items or [],  # Pass empty list to maintain compatibility
        )
        
        # CRITICAL: Force CSV ingestion for manual mode
        # The legacy UI path is always CSV-based, so we override any guessing
        eval_pack.ingestion.type = "csv"
        eval_pack.ingestion.config = {"mode": mode.value}

    # ------------------------------------------------------------------
    # 2. Handle Generation (Mode B) if specified in the pack
    # ------------------------------------------------------------------
    # FIX: Add the branching logic for Mode B generation when using an Eval Pack.
    if eval_pack.generation and raw_data is not None:
        if not user_context:
            raise ValueError("User context is required for generation but was not provided.")
        
        logger.info("Pack has a 'generation' block. Running Mode B workflow.")
        
        if eval_pack.ingestion.type != "csv":
            # Use the pack's declared ingester (e.g., 'python') to build items first
            try:
                ingester_cls = ComponentRegistry.get_ingester(eval_pack.ingestion.type)
                ingester = ingester_cls()
                ingestion_cfg = eval_pack.ingestion.config or {}
                items = ingester.ingest(raw_data, ingestion_cfg)
                logger.info(f"Ingested {len(items)} items via '{eval_pack.ingestion.type}' for generation.")
            except Exception as e:
                logger.error(f"Data ingestion failed in Mode B (non-CSV): {e}", exc_info=True)
                raise ValueError(f"Data ingestion failed: {e}")

            # Now run Mode B generation from items (no CSV assumption)
            items, generation_metadata = await handle_mode_b_generation_from_items(
                items=items,
                generation_config=eval_pack.generation,
                user_context=user_context,
                api_keys=api_keys or {},
                progress_callback=progress_callback
            )
        else:
            # Keep legacy CSV path unchanged
            items, generation_metadata = await handle_mode_b_generation(
                raw_data=raw_data,
                generation_config=eval_pack.generation,
                user_context=user_context,
                api_keys=api_keys or {},
                progress_callback=progress_callback
            )
        
        # Prevent re-ingestion later in the pipeline
        raw_data = None
        
        # If the goal was just to create a dataset, stop here and return the results.
        if eval_pack.generation.mode == GenerationMode.GENERATE_EXPECTED_OUTPUTS:
            logger.info("Mode B 'Generate Expected Outputs' complete. Returning generated data without scoring.")
            return EvaluationResults(
                items=items,
                config={"eval_pack": eval_pack.model_dump(mode='json'), **generation_metadata},
                metadata={"mode": "generate_expected_outputs", "total_items": len(items)}
            )
        
        # Otherwise, the generated items will proceed to the scoring pipeline.
        logger.info("Mode B 'Generate Outputs' complete. Proceeding to evaluation.")

    # ------------------------------------------------------------------
    # 3. Ingest raw data if it hasn't been generated
    # ------------------------------------------------------------------
    if raw_data is not None:
        logger.info("Ingesting raw data before evaluation")

        ingester_type = eval_pack.ingestion.type
        logger.info(f'Using ingester "{ingester_type}" from pack')

        try:
            ingester_cls = ComponentRegistry.get_ingester(ingester_type)
            ingester = ingester_cls()
        except ValueError as e:
            raise ValueError(
                f"Pack specifies unknown ingester '{ingester_type}'. "
                f"Available ingesters: {list(ComponentRegistry._ingesters.keys())}"
            ) from e

        try:
            ingestion_cfg = eval_pack.ingestion.config or {}
            items = ingester.ingest(raw_data, ingestion_cfg)
        except Exception as e:
            logger.error(f"Ingestion failed: {e}")
            raise ValueError(f"Failed to ingest data using {ingester_type}: {e}") from e

        logger.info("Ingested %d items", len(items))

    elif items is None:
        raise ValueError("Either 'raw_data' or 'items' must be provided")

    # ------------------------------------------------------------------
    # 4. Delegate to the scoring implementation
    # ------------------------------------------------------------------
    return await _run_evaluation_batch_impl(
        items=items,
        selected_scorers=selected_scorers,
        scorer_configs=scorer_configs,
        api_keys=api_keys,
        batch_size=batch_size,
        progress_callback=progress_callback,
        pack=eval_pack,
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