"""
Pipeline executor for running Eval Pack pipelines.

This file contains the core logic for taking a list of EvaluationItems and
an EvalPack, and then executing each stage of the pack's pipeline against
each item.
"""

import asyncio
import logging
import os
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from core.data_models import EvaluationItem, ScorerResult, EvaluationResults
from core.eval_pack.schema import EvalPackV1, PipelineStage
from core.registry import ComponentRegistry
from core.utils.tracing import get_tracer

logger = logging.getLogger(__name__)


class PipelineExecutor:
    """Executes evaluation pipelines defined in Eval Packs."""

    # This dictionary maps OpenInference environment variables to the fields
    # they control, allowing for privacy masking of sensitive data in traces.
    PRIVACY_ENV_VARS = {
        "OPENINFERENCE_HIDE_INPUTS": "input",
        "OPENINFERENCE_HIDE_OUTPUTS": "output",
        "OPENINFERENCE_HIDE_INPUT_MESSAGES": "input.messages",
        "OPENINFERENCE_HIDE_OUTPUT_MESSAGES": "output.messages",
        "OPENINFERENCE_HIDE_INPUT_IMAGES": "input.images",
        "OPENINFERENCE_HIDE_INPUT_TOOLS": "input.tools",
        "OPENINFERENCE_HIDE_TOOL_PARAMETERS": "tool.parameters",
        "OPENINFERENCE_HIDE_EMBEDDING_EMBEDDINGS": "embedding.embeddings",
        "OPENINFERENCE_HIDE_EMBEDDING_TEXT": "embedding.text",
        "OPENINFERENCE_HIDE_LLM_TOKEN_COUNT_PROMPT": "llm.token_count.prompt",
        "OPENINFERENCE_HIDE_LLM_TOKEN_COUNT_COMPLETION": "llm.token_count.completion",
        "OPENINFERENCE_BASE64_IMAGE_MAX_LENGTH": "image.base64.max_length",
    }

    def __init__(self, eval_pack: EvalPackV1):
        """
        Initialize the executor with an Eval Pack.
        """
        self.eval_pack = eval_pack
        self._tracer = get_tracer(__name__)
        self._privacy_settings = self._load_privacy_settings()
        
        ### ARCHITECTURAL CHANGE (per Dia's advice) ###
        # The `self._scorers` cache has been REMOVED. Previously, a single instance
        # of each scorer was created and cached here. This was the root cause of
        # configuration from one stage "bleeding" into another. Scorers will now
        # be instantiated on-the-fly within the `run_batch` loop, making them stateless.

    def _load_privacy_settings(self) -> Dict[str, bool]:
        """Load privacy settings from environment variables."""
        settings = {}
        for env_var, attribute in self.PRIVACY_ENV_VARS.items():
            value = os.environ.get(env_var, "").lower()
            settings[attribute] = value in ("true", "1", "yes")
        return settings

    def _mask_sensitive_data(self, item: EvaluationItem) -> EvaluationItem:
        """Mask sensitive data in an evaluation item based on privacy settings."""
        if not any(self._privacy_settings.values()):
            return item

        # Create a copy to avoid modifying the original
        masked_item = item.model_copy(deep=True)

        # Mask input/output if requested
        if self._privacy_settings.get("input", False):
            masked_item.input = "[MASKED]"
        if self._privacy_settings.get("output", False) and masked_item.output:
            masked_item.output = "[MASKED]"

        # Mask metadata fields based on settings
        for attr_path, should_mask in self._privacy_settings.items():
            if should_mask and "." in attr_path:
                parts = attr_path.split(".")
                if parts[0] in masked_item.metadata:
                    self._mask_nested_attribute(masked_item.metadata, parts, "[MASKED]")

        return masked_item

    def _mask_nested_attribute(self, obj: Dict[str, Any], path: List[str], mask_value: Any):
        """Recursively mask nested attributes in a dictionary."""
        if len(path) == 1:
            if path[0] in obj:
                obj[path[0]] = mask_value
        elif path[0] in obj and isinstance(obj[path[0]], dict):
            self._mask_nested_attribute(obj[path[0]], path[1:], mask_value)

    async def initialize(self, api_keys: Optional[Dict[str, str]] = None):
        """
        Initialize the executor by storing API keys for the run.
        Scorers are no longer instantiated here to ensure they remain stateless.
        """
        self.api_keys = api_keys or {}
        logger.info(f"Pipeline Executor initialized for pack: '{self.eval_pack.name}'")

    async def run_batch(
        self,
        items: List[EvaluationItem],
        batch_size: int = 10,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> EvaluationResults:
        """
        Run the full evaluation pipeline for a list of items.

        This is the main execution engine. It iterates through each item and,
        for each item, iterates through each stage in the Eval Pack's pipeline,
        handling conditional execution, scorer instantiation, and error handling.
        """
        start_time = datetime.now()
        
        with self._tracer.start_as_current_span("pipeline_batch_execution") as span:
            span.set_attribute("eval_pack.name", self.eval_pack.name)
            span.set_attribute("items.count", len(items))

            for item in items:
                # Ensure scores from any previous runs are cleared.
                item.scores = []
                
                for stage in self.eval_pack.pipeline:
                    
                    ### FIX #2: IMPLEMENT `run_if` CONDITIONAL EXECUTION ###
                    if stage.run_if:
                        try:
                            ### DIA'S SUGGESTION: DEFENSIVE PROGRAMMING for `eval()` ###
                            # NOTE: Using eval() on untrusted input is a security risk.
                            # This is acceptable for now as the Eval Pack YAML is controlled
                            # by the user. For a production system handling untrusted packs,
                            # replace this with a safer expression evaluator like `asteval`.
                            context = {"metadata": item.metadata, "input": item.input, "output": item.output}
                            if not eval(stage.run_if, {"__builtins__": {}}, context):
                                logger.debug(f"Skipping stage '{stage.name}' for item '{item.id}' due to run_if condition.")
                                continue # Skip to the next stage
                        except Exception as e:
                            logger.warning(f"Could not evaluate run_if condition '{stage.run_if}' for stage '{stage.name}'. Skipping stage. Error: {e}")
                            continue # Gracefully skip stage on evaluation error

                    try:
                        ### FIX #1 (CORE ARCHITECTURE): INSTANTIATE SCORER ON-THE-FLY ###
                        # This ensures the scorer is stateless and uses the correct
                        # configuration for the current stage, preventing config bleeding.
                        scorer_class = ComponentRegistry.get_scorer(stage.scorer)
                        stage_config = stage.config.copy()

                        # Inject API key into the stage config if the scorer class requires it.
                        if scorer_class.requires_api_key and "api_key" not in stage_config:
                            provider = stage_config.get("provider", "openai")
                            if provider in self.api_keys:
                                stage_config["api_key"] = self.api_keys[provider]

                        scorer_instance = scorer_class()

                        # Mask data if privacy settings are enabled
                        scoring_item = self._mask_sensitive_data(item)

                        ### FIX #1 (CORE ARCHITECTURE): PASS STAGE_CONFIG TO SCORE METHOD ###
                        # The scorer's score method now receives the stage-specific config.
                        if asyncio.iscoroutinefunction(scorer_instance.score):
                            result = await scorer_instance.score(scoring_item, stage_config)
                        else:
                            # Allow for non-async scorers
                            result = scorer_instance.score(scoring_item, stage_config)

                        item.scores.append(result)

                        # Handle the 'on_fail' behavior for the stage
                        if not result.passed and stage.on_fail == "stop":
                            logger.info(f"Stopping pipeline for item {item.id} due to failure in stage {stage.name}")
                            break  # Stop processing further stages for this item

                    except Exception as e:
                        # DIA'S SUGGESTION: Robust error handling
                        # If a scorer fails unexpectedly, log it, record an error result,
                        # and continue or stop based on the on_fail policy.
                        logger.error(f"Error executing stage '{stage.name}' for item '{item.id}': {e}", exc_info=True)
                        error_result = ScorerResult(
                            scorer_name=stage.scorer,
                            score=0.0,
                            passed=False,
                            error=str(e),
                            reasoning="An unexpected error occurred during the scoring process."
                        )
                        item.scores.append(error_result)
                        if stage.on_fail == "stop":
                            break
        
        # Assemble and return the final results object
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        results = EvaluationResults(
            items=items,
            config={
                "eval_pack": self.eval_pack.model_dump(mode='json'),
                "batch_size": batch_size,
                "privacy_settings": {k: v for k, v in self._privacy_settings.items() if v},
            },
            metadata={
                "execution_time_seconds": duration,
                "start_time_utc": start_time.isoformat(),
                "end_time_utc": end_time.isoformat(),
                "total_items": len(items),
                "total_stages": len(self.eval_pack.pipeline),
                "eval_pack_metadata": self.eval_pack.metadata,
            }
        )
        
        results.calculate_summary_stats()
        return results