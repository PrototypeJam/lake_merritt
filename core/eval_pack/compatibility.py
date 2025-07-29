"""
Compatibility layer for translating legacy UI selections into Eval Pack format.
"""
import copy # <-- FIX: Import the copy module for deepcopy
import logging
from typing import Any, Dict, List, Optional

from core.data_models import EvaluationItem
from core.eval_pack.schema import (
    EvalPackV1,
    IngestionConfig,
    PipelineStage,
    SchemaVersion,
)

logger = logging.getLogger(__name__)


def create_legacy_pack(
    selected_scorers: List[str],
    scorer_configs: Dict[str, Dict[str, Any]],
    api_keys: Dict[str, str],
    items: Optional[List[EvaluationItem]] = None,
) -> EvalPackV1:
    """
    Create an in-memory Eval Pack from legacy UI selections.
    
    This function translates the traditional scorer selections and configurations
    into the new Eval Pack format, allowing backward compatibility with existing
    UI workflows.
    
    Args:
        selected_scorers: List of scorer names selected in the UI
        scorer_configs: Configuration for each scorer
        api_keys: API keys for LLM providers
        items: Optional list of evaluation items (used to determine ingestion type)
    
    Returns:
        An EvalPackV1 instance representing the legacy configuration
    """
    logger.info(f"Creating legacy pack with scorers: {selected_scorers}")
    
    # Determine ingestion type based on provided items
    if items and len(items) > 0:
        # Check if items look like they came from CSV (simple structure)
        first_item = items[0]
        if (hasattr(first_item, 'metadata') and 
            isinstance(first_item.metadata, dict) and
            len(first_item.metadata) <= 2):  # Simple metadata suggests CSV
            ingestion_type = "csv"
        else:
            # Default to JSON for more complex structures
            ingestion_type = "json"
    else:
        # Default to JSON if no items provided
        ingestion_type = "json"
    
    # Create ingestion config
    ingestion_config = IngestionConfig(
        type=ingestion_type,
        config={}  # Legacy mode doesn't need specific ingestion config
    )
    
    # Create pipeline stages from selected scorers
    pipeline_stages = []
    for scorer_name in selected_scorers:
        # FIX: Use a deepcopy to ensure nested configs (like LLM Judge prompts) are preserved.
        config = copy.deepcopy(scorer_configs.get(scorer_name, {}))
        
        # Handle API keys for LLM-based scorers
        if scorer_name in ["llm_judge", "criteria_selection_judge"]:
            if "api_key" not in config:
                provider = config.get("provider", "openai")
                api_key = api_keys.get(provider)
                if api_key:
                    config["api_key"] = api_key
        
        # Create pipeline stage
        stage = PipelineStage(
            name=f"{scorer_name}_stage",
            scorer=scorer_name,
            config=config,
            on_fail="continue"  # Legacy behavior: continue on failure
        )
        pipeline_stages.append(stage)
    
    # Create the Eval Pack
    eval_pack = EvalPackV1(
        schema_version=SchemaVersion.V1_0,
        name="Legacy UI Configuration",
        version="1.0",
        description="Automatically generated from legacy UI selections",
        author="Legacy UI",
        ingestion=ingestion_config,
        pipeline=pipeline_stages,
        metadata={
            "source": "legacy_ui",
            "auto_generated": True,
            "selected_scorers": selected_scorers,
        }
    )
    
    logger.info(f"Created legacy pack with {len(pipeline_stages)} stages")
    return eval_pack


def extract_scorer_configs(eval_pack: EvalPackV1) -> Dict[str, Dict[str, Any]]:
    """
    Extract scorer configurations from an Eval Pack for use with legacy code.
    
    Args:
        eval_pack: The Eval Pack to extract configurations from
    
    Returns:
        A dictionary mapping scorer names to their configurations
    """
    scorer_configs = {}
    for stage in eval_pack.pipeline:
        scorer_configs[stage.scorer] = stage.config.copy()
    return scorer_configs


def extract_selected_scorers(eval_pack: EvalPackV1) -> List[str]:
    """
    Extract the list of selected scorers from an Eval Pack.
    
    Args:
        eval_pack: The Eval Pack to extract scorers from
    
    Returns:
        A list of scorer names in pipeline order
    """
    return [stage.scorer for stage in eval_pack.pipeline]