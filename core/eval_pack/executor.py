"""
Pipeline executor for running Eval Pack pipelines.
"""

import asyncio
import logging
import os
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from core.data_models import EvaluationItem, ScorerResult, EvaluationResults
from core.eval_pack.schema import EvalPackV1, PipelineStage, SpanKind
from core.registry import ComponentRegistry
from core.scoring.base import BaseScorer
from core.utils.tracing import get_tracer

# OpenInference semantic conventions
from openinference.semconv.trace import SpanAttributes

logger = logging.getLogger(__name__)


class PipelineExecutor:
    """Executes evaluation pipelines defined in Eval Packs."""
    
    # OpenInference privacy environment variables
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
        
        Args:
            eval_pack: The Eval Pack containing the pipeline definition
        """
        self.eval_pack = eval_pack
        self._scorers: Dict[str, BaseScorer] = {}
        self._tracer = get_tracer(__name__)
        self._privacy_settings = self._load_privacy_settings()
    
    def _load_privacy_settings(self) -> Dict[str, bool]:
        """Load privacy settings from environment variables."""
        settings = {}
        for env_var, attribute in self.PRIVACY_ENV_VARS.items():
            value = os.environ.get(env_var, "").lower()
            settings[attribute] = value in ("true", "1", "yes")
        return settings
    
    def _mask_sensitive_data(self, item: EvaluationItem) -> EvaluationItem:
        """Mask sensitive data in evaluation item based on privacy settings."""
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
                # Handle nested attributes in metadata
                parts = attr_path.split(".")
                if parts[0] in masked_item.metadata:
                    self._mask_nested_attribute(masked_item.metadata, parts, "[MASKED]")
        
        return masked_item
    
    def _mask_nested_attribute(self, obj: Dict[str, Any], path: List[str], mask_value: Any):
        """Recursively mask nested attributes in a dictionary."""
        if len(path) == 1:
            if path[0] in obj:
                obj[path[0]] = mask_value
        else:
            if path[0] in obj and isinstance(obj[path[0]], dict):
                self._mask_nested_attribute(obj[path[0]], path[1:], mask_value)
        
    async def initialize(self, api_keys: Optional[Dict[str, str]] = None):
        """
        Initialize all scorers in the pipeline.
        
        Args:
            api_keys: Optional API keys for LLM-based scorers
        """
        api_keys = api_keys or {}
        
        with self._tracer.start_as_current_span("pipeline_initialization") as span:
            span.set_attribute("eval_pack.name", self.eval_pack.name)
            span.set_attribute("eval_pack.version", self.eval_pack.version)
            span.set_attribute("pipeline.stages", len(self.eval_pack.pipeline))
            
            for stage in self.eval_pack.pipeline:
                if stage.scorer not in self._scorers:
                    config = stage.config.copy()
                    
                    # Get scorer class to check if it requires API key
                    scorer_class = ComponentRegistry.get_scorer(stage.scorer)
                    
                    # Inject API key if scorer requires it and it's not already in config
                    if hasattr(scorer_class, 'requires_api_key') and scorer_class.requires_api_key:
                        if "api_key" not in config:
                            provider = config.get("provider", "openai")
                            if provider in api_keys:
                                config["api_key"] = api_keys[provider]
                            else:
                                logger.warning(
                                    f"Scorer {stage.scorer} requires API key for provider {provider}, "
                                    f"but none was provided"
                                )
                    
                    # Instantiate scorer
                    self._scorers[stage.scorer] = scorer_class(config)
                    logger.info(f"Initialized scorer: {stage.scorer}")
    
    async def run(
        self,
        items: List[EvaluationItem],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> EvaluationResults:
        """
        Run the pipeline on a list of evaluation items.
        
        Args:
            items: List of evaluation items to process
            progress_callback: Optional callback for progress updates
            
        Returns:
            EvaluationResults with scored items and metadata
        """
        start_time = datetime.now()
        total_operations = len(items) * len(self.eval_pack.pipeline)
        completed_operations = 0
        
        with self._tracer.start_as_current_span("pipeline_execution") as span:
            span.set_attribute("eval_pack.name", self.eval_pack.name)
            span.set_attribute("eval_pack.version", self.eval_pack.version)
            span.set_attribute("items.count", len(items))
            span.set_attribute("pipeline.stages", len(self.eval_pack.pipeline))
            
            for item in items:
                # Clear existing scores
                item.scores = []
                
                # Run each stage in the pipeline
                for stage in self.eval_pack.pipeline:
                    with self._tracer.start_as_current_span(f"stage_{stage.name}") as stage_span:
                        stage_span.set_attribute("stage.name", stage.name)
                        stage_span.set_attribute("stage.scorer", stage.scorer)
                        if stage.span_kind:
                            stage_span.set_attribute("stage.span_kind", stage.span_kind.value)
                        
                        # Check if we should run this stage based on span_kind
                        if stage.span_kind:
                            # Check if item has the required span kind in metadata
                            item_span_kind = item.metadata.get("span_kind")
                            if item_span_kind != stage.span_kind.value:
                                logger.debug(
                                    f"Skipping stage {stage.name} for item {item.id}: "
                                    f"span_kind mismatch (expected {stage.span_kind.value}, got {item_span_kind})"
                                )
                                stage_span.set_attribute("stage.skipped", True)
                                stage_span.set_attribute("stage.skip_reason", "span_kind_mismatch")
                                completed_operations += 1
                                if progress_callback:
                                    progress_callback(completed_operations, total_operations)
                                continue
                        
                        try:
                            # Get the scorer
                            scorer = self._scorers[stage.scorer]
                            
                            # Mask sensitive data if privacy settings are enabled
                            scoring_item = self._mask_sensitive_data(item)
                            
                            # Run scorer
                            if asyncio.iscoroutinefunction(scorer.score):
                                result = await scorer.score(scoring_item)
                            else:
                                result = scorer.score(scoring_item)
                            
                            item.scores.append(result)
                            logger.debug(
                                f"Stage {stage.name} scored item {item.id}: {result.score}"
                            )
                            
                            stage_span.set_attribute("score.value", str(result.score))
                            stage_span.set_attribute("score.passed", result.passed)
                            
                            # Check if we should stop on failure
                            if not result.passed and stage.on_fail == "stop":
                                logger.info(
                                    f"Stopping pipeline for item {item.id} due to failure in stage {stage.name}"
                                )
                                stage_span.set_attribute("pipeline.stopped", True)
                                break
                                
                        except Exception as e:
                            logger.error(f"Error in stage {stage.name} for item {item.id}: {e}")
                            stage_span.record_exception(e)
                            stage_span.set_status("error")
                            
                            # Add error result
                            error_result = ScorerResult(
                                scorer_name=stage.scorer,
                                score=0.0,
                                passed=False,
                                error=str(e),
                            )
                            item.scores.append(error_result)
                            
                            # Check if we should stop on failure
                            if stage.on_fail == "stop":
                                logger.info(
                                    f"Stopping pipeline for item {item.id} due to error in stage {stage.name}"
                                )
                                stage_span.set_attribute("pipeline.stopped", True)
                                break
                        
                        completed_operations += 1
                        if progress_callback:
                            progress_callback(completed_operations, total_operations)
        
        # Create EvaluationResults
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        results = EvaluationResults(
            items=items,
            config={
                "eval_pack": {
                    "name": self.eval_pack.name,
                    "version": self.eval_pack.version,
                    "description": self.eval_pack.description,
                    "author": self.eval_pack.author,
                },
                "pipeline": [
                    {
                        "name": stage.name,
                        "scorer": stage.scorer,
                        "config": stage.config,
                        "on_fail": stage.on_fail,
                        "span_kind": stage.span_kind.value if stage.span_kind else None,
                    }
                    for stage in self.eval_pack.pipeline
                ],
                "privacy_settings": {
                    k: v for k, v in self._privacy_settings.items() if v
                },
            },
            metadata={
                "execution_time": duration,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "total_items": len(items),
                "total_stages": len(self.eval_pack.pipeline),
                "eval_pack_metadata": self.eval_pack.metadata,
            }
        )
        
        # Calculate summary statistics
        results.calculate_summary_stats()
        
        return results
    
    async def run_batch(
        self,
        items: List[EvaluationItem],
        batch_size: int = 10,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> EvaluationResults:
        """
        Run the pipeline in batches for better performance with async scorers.
        
        Args:
            items: List of evaluation items to process
            batch_size: Number of items to process in parallel
            progress_callback: Optional callback for progress updates
            
        Returns:
            EvaluationResults with scored items and metadata
        """
        start_time = datetime.now()
        total_operations = len(items) * len(self.eval_pack.pipeline)
        completed_operations = 0
        
        with self._tracer.start_as_current_span("pipeline_batch_execution") as span:
            span.set_attribute("eval_pack.name", self.eval_pack.name)
            span.set_attribute("eval_pack.version", self.eval_pack.version)
            span.set_attribute("items.count", len(items))
            span.set_attribute("batch.size", batch_size)
            span.set_attribute("pipeline.stages", len(self.eval_pack.pipeline))
            
            # Process items in batches
            for i in range(0, len(items), batch_size):
                batch = items[i : i + batch_size]
                
                with self._tracer.start_as_current_span(f"batch_{i//batch_size}") as batch_span:
                    batch_span.set_attribute("batch.index", i // batch_size)
                    batch_span.set_attribute("batch.items", len(batch))
                    
                    # For each stage in the pipeline
                    for stage_idx, stage in enumerate(self.eval_pack.pipeline):
                        with self._tracer.start_as_current_span(f"stage_{stage.name}") as stage_span:
                            stage_span.set_attribute("stage.name", stage.name)
                            stage_span.set_attribute("stage.scorer", stage.scorer)
                            if stage.span_kind:
                                stage_span.set_attribute("stage.span_kind", stage.span_kind.value)
                            
                            # Create tasks for this batch and stage
                            tasks = []
                            
                            for item in batch:
                                # Initialize scores list if this is the first stage
                                if stage_idx == 0:
                                    item.scores = []
                                
                                # Check span_kind filtering
                                if stage.span_kind:
                                    item_span_kind = item.metadata.get("span_kind")
                                    if item_span_kind != stage.span_kind.value:
                                        completed_operations += 1
                                        if progress_callback:
                                            progress_callback(completed_operations, total_operations)
                                        continue
                                
                                # Get the scorer
                                scorer = self._scorers[stage.scorer]
                                
                                # Mask sensitive data if privacy settings are enabled
                                scoring_item = self._mask_sensitive_data(item)
                                
                                # Create task
                                if asyncio.iscoroutinefunction(scorer.score):
                                    task = scorer.score(scoring_item)
                                else:
                                    task = asyncio.create_task(asyncio.to_thread(scorer.score, scoring_item))
                                
                                tasks.append((item, stage, task))
                            
                            # Wait for all tasks in this batch/stage to complete
                            for item, stage, task in tasks:
                                try:
                                    result = await task
                                    item.scores.append(result)
                                    
                                    # Check if we should stop on failure
                                    if not result.passed and stage.on_fail == "stop":
                                        logger.info(
                                            f"Stage {stage.name} failed for item {item.id}, "
                                            f"but continuing with other items in batch"
                                        )
                                        
                                except Exception as e:
                                    logger.error(f"Error in stage {stage.name} for item {item.id}: {e}")
                                    stage_span.record_exception(e)
                                    
                                    # Add error result
                                    error_result = ScorerResult(
                                        scorer_name=stage.scorer,
                                        score=0.0,
                                        passed=False,
                                        error=str(e),
                                    )
                                    item.scores.append(error_result)
                                
                                completed_operations += 1
                                if progress_callback:
                                    progress_callback(completed_operations, total_operations)
        
        # Create EvaluationResults
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        results = EvaluationResults(
            items=items,
            config={
                "eval_pack": {
                    "name": self.eval_pack.name,
                    "version": self.eval_pack.version,
                    "description": self.eval_pack.description,
                    "author": self.eval_pack.author,
                },
                "pipeline": [
                    {
                        "name": stage.name,
                        "scorer": stage.scorer,
                        "config": stage.config,
                        "on_fail": stage.on_fail,
                        "span_kind": stage.span_kind.value if stage.span_kind else None,
                    }
                    for stage in self.eval_pack.pipeline
                ],
                "batch_size": batch_size,
                "privacy_settings": {
                    k: v for k, v in self._privacy_settings.items() if v
                },
            },
            metadata={
                "execution_time": duration,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "total_items": len(items),
                "total_stages": len(self.eval_pack.pipeline),
                "batch_size": batch_size,
                "eval_pack_metadata": self.eval_pack.metadata,
            }
        )
        
        # Calculate summary statistics
        results.calculate_summary_stats()
        
        return results