"""
Logic for generating outputs from an Actor LLM (Mode B).
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from core.data_models import EvaluationItem, LLMConfig
from services.llm_clients import create_llm_client

logger = logging.getLogger(__name__)


async def generate_outputs(
    items: List[EvaluationItem],
    actor_config: Dict[str, Any],
    progress_callback: Optional[Callable[[int, int], None]] = None,
    batch_size: int = 5,
) -> List[EvaluationItem]:
    """
    Generate outputs for evaluation items using an Actor LLM.

    Args:
        items: List of evaluation items (with input and expected_output)
        actor_config: Configuration for the Actor LLM
        progress_callback: Optional callback for progress updates
        batch_size: Number of items to process concurrently

    Returns:
        List of evaluation items with generated outputs
    """
    start_time = datetime.now()
    logger.info(f"Starting output generation for {len(items)} items")

    # Create LLM client
    client = create_llm_client(
        provider=actor_config["provider"],
        api_key=actor_config.get("api_key"),
    )

    # Prepare generation parameters
    gen_params = {
        "model": actor_config["model"],
        "temperature": actor_config.get("temperature", 0.7),
        "max_tokens": actor_config.get("max_tokens", 1000),
    }

    system_prompt = actor_config.get("system_prompt")

    # Process items in batches
    total_items = len(items)
    completed_items = 0

    for i in range(0, len(items), batch_size):
        batch = items[i : i + batch_size]
        tasks = []

        for item in batch:
            # Prepare the prompt
            if system_prompt:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": item.input},
                ]
            else:
                messages = [{"role": "user", "content": item.input}]

            # Create generation task
            task = client.generate(messages, **gen_params)
            tasks.append((item, task))

        # Wait for batch to complete
        for item, task in tasks:
            try:
                output = await task
                item.output = output
                logger.debug(f"Generated output for item {item.id or 'unknown'}")
            except Exception as e:
                logger.error(
                    f"Error generating output for item {item.id or 'unknown'}: {e}"
                )
                item.output = f"[ERROR: Failed to generate output - {str(e)}]"

            completed_items += 1
            if progress_callback:
                progress_callback(completed_items, total_items)

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    logger.info(f"Output generation completed in {duration:.2f} seconds")
    return items


async def generate_single_output(
    input_text: str,
    actor_config: Dict[str, Any],
) -> str:
    """
    Generate a single output for testing or one-off generation.

    Args:
        input_text: The input prompt
        actor_config: Configuration for the Actor LLM

    Returns:
        Generated output text
    """
    client = create_llm_client(
        provider=actor_config["provider"],
        api_key=actor_config.get("api_key"),
    )

    gen_params = {
        "model": actor_config["model"],
        "temperature": actor_config.get("temperature", 0.7),
        "max_tokens": actor_config.get("max_tokens", 1000),
    }

    system_prompt = actor_config.get("system_prompt")

    if system_prompt:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": input_text},
        ]
    else:
        messages = [{"role": "user", "content": input_text}]

    try:
        output = await client.generate(messages, **gen_params)
        return output
    except Exception as e:
        logger.error(f"Error generating output: {e}")
        raise


def validate_generation_config(config: Dict[str, Any]) -> bool:
    """
    Validate that a generation configuration has all required fields.

    Args:
        config: Configuration dictionary

    Returns:
        True if valid, raises ValueError if not
    """
    required_fields = ["provider", "model"]

    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required field in generation config: {field}")

    # Validate provider
    valid_providers = ["openai", "anthropic", "google"]
    if config["provider"] not in valid_providers:
        raise ValueError(
            f"Invalid provider: {config['provider']}. Must be one of {valid_providers}"
        )

    # Validate model based on provider
    provider_models = {
        "openai": ["gpt-4", "gpt-4-turbo-preview", "gpt-3.5-turbo"],
        "anthropic": [
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
        ],
        "google": ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-1.0-pro"],
    }

    valid_models = provider_models.get(config["provider"], [])
    if config["model"] not in valid_models:
        logger.warning(
            f"Model {config['model']} not in known models for {config['provider']}"
        )

    return True
