# In file: core/generation_handler.py

"""
Dedicated handler for Mode B generation workflows.
Keeps generation logic separate from evaluation.
"""

import logging
from typing import Dict, Any, List, Optional, Union, IO, Tuple, Callable
import pandas as pd

from core.data_models import EvaluationItem
from core.eval_pack.schema import GenerationConfig, GenerationMode
from core.generation import create_generation_prompt, generate_data_for_items
from core.ingestion.csv_ingester import CSVIngester

logger = logging.getLogger(__name__)


async def handle_mode_b_generation(
    raw_data: Union[IO, pd.DataFrame],
    generation_config: GenerationConfig,
    user_context: str,
    api_keys: Dict[str, str],
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> Tuple[List[EvaluationItem], Dict[str, Any]]:
    """
    Handle the complete Mode B generation workflow.
    
    Args:
        raw_data: Input CSV data.
        generation_config: Configuration for the generation process.
        user_context: User-provided high-level context.
        api_keys: Dictionary of API keys for LLM providers.
        progress_callback: Optional callback for progress updates.
        
    Returns:
        A tuple containing the list of generated items and metadata about the run.
    """
    logger.info(f"Starting Mode B generation in mode: {generation_config.mode.value}")
    
    # Step 1: Ingest the raw data. The ingester is simple and stateless.
    ingester = CSVIngester()
    try:
        items = ingester.ingest(raw_data, {})
        logger.info(f"Successfully ingested {len(items)} items for generation.")
    except Exception as e:
        logger.error(f"Data ingestion failed: {e}", exc_info=True)
        raise ValueError(f"Data ingestion failed: {str(e)}")

    # Step 2: Perform workflow-specific validation *after* ingestion.
    # This logic is now correctly placed in the handler, not the ingester.
    if generation_config.mode == GenerationMode.GENERATE_OUTPUTS:
        missing_expected = [item.id for item in items if item.expected_output is None]
        if missing_expected:
            raise ValueError(
                f"For 'Generate Outputs' mode, all rows in the CSV must have an 'expected_output' column. "
                f"Missing for items: {', '.join(missing_expected[:5])}{'...' if len(missing_expected) > 5 else ''}"
            )
            
    # Step 3: Generate the system prompt (with optional meta-prompting)
    try:
        system_prompt = await create_generation_prompt(generation_config, user_context, api_keys)
        logger.info("Successfully created system prompt for data generation.")
    except Exception as e:
        logger.error(f"Prompt generation failed: {e}", exc_info=True)
        raise ValueError(f"Prompt generation failed: {str(e)}")
    
    # Step 4: Initialize metadata for the generation run
    generation_metadata = {
        "total_items": len(items),
        "mode": generation_config.mode.value,
        "used_meta_prompting": generation_config.use_meta_prompting,
        "successful_generations": 0,
        "failed_generations": 0,
    }
    
    # Step 5: Generate the data for each item
    try:
        items = await generate_data_for_items(
            items, 
            system_prompt, 
            generation_config,
            api_keys,
            progress_callback=progress_callback
        )
        
        # Tally successes and failures after generation
        for item in items:
            field_to_check = item.output if generation_config.mode == GenerationMode.GENERATE_OUTPUTS else item.expected_output
            if field_to_check and not field_to_check.startswith("[ERROR"):
                generation_metadata["successful_generations"] += 1
            else:
                generation_metadata["failed_generations"] += 1
                    
    except Exception as e:
        logger.error(f"The generation process failed: {e}", exc_info=True)
        raise ValueError(f"Data generation failed: {str(e)}")
    
    logger.info(f"Generation complete. Success: {generation_metadata['successful_generations']}, Failures: {generation_metadata['failed_generations']}")
    return items, generation_metadata


def prepare_csv_for_download(
    items: List[EvaluationItem]
) -> str:
    """
    Prepare the generated data as a CSV string for download.
    Dynamically includes columns based on what data is present.
    """
    if not items:
        return ""

    # Determine which optional columns are present in the data
    has_output = any(item.output is not None for item in items)
    has_expected_output = any(item.expected_output is not None for item in items)
    
    data = []
    for item in items:
        row = {"input": item.input}
        if has_output:
            row["output"] = item.output or ""
        if has_expected_output:
            row["expected_output"] = item.expected_output or ""
        
        for key, value in item.metadata.items():
            row[f"metadata_{key}"] = str(value)
        data.append(row)
    
    df = pd.DataFrame(data)
    return df.to_csv(index=False)