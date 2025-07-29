# In file: core/generation.py

"""
Enhanced generation module for Mode B functionality.
"""

import asyncio
import logging
from typing import Any, Callable, Dict, List, Optional
import jinja2

from core.data_models import EvaluationItem
from core.eval_pack.schema import GenerationConfig, GenerationMode, LLMConfig
from services.llm_clients import create_llm_client

logger = logging.getLogger(__name__)


async def create_generation_prompt(config: GenerationConfig, user_context: str, api_keys: Dict[str, str]) -> str:
    """
    Meta-prompting step: Use an LLM to generate the final system prompt.
    If not configured or disabled, just use the user-provided context directly.
    """
    if config.use_meta_prompting and config.prompt_creation_template and config.prompt_generator_llm:
        logger.info("Using meta-prompting to generate system prompt.")
        template = jinja2.Template(config.prompt_creation_template)
        meta_prompt = template.render(context=user_context, mode=config.mode.value)
        
        generator_llm_config = config.prompt_generator_llm
        if not generator_llm_config.api_key:
            generator_llm_config.api_key = api_keys.get(generator_llm_config.provider)

        client = create_llm_client(generator_llm_config.provider, generator_llm_config.api_key)
        
        try:
            prompt = await client.generate(
                [{"role": "system", "content": "You are an expert prompt engineer."},
                 {"role": "user", "content": meta_prompt}],
                model=generator_llm_config.model,
                temperature=generator_llm_config.temperature,
                max_tokens=generator_llm_config.max_tokens,
            )
            logger.info("Successfully generated system prompt via meta-prompting.")
            return prompt.strip()
        except Exception as e:
            logger.error(f"Meta-prompting failed: {e}. Falling back to using context directly.")
            return user_context
    else:
        return user_context


async def _generate_for_single_item(
    item: EvaluationItem,
    system_prompt: str,
    template: jinja2.Template,
    client: Any,
    generator_config: LLMConfig
) -> str:
    """Helper function to run generation for one item, allowing exceptions to be caught by gather."""
    try:
        user_prompt = template.render(
            item=item.model_dump(),
            context=system_prompt
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        result = await client.generate(
            messages,
            model=generator_config.model,
            temperature=generator_config.temperature,
            max_tokens=generator_config.max_tokens,
        )
        if not result or not result.strip():
            raise ValueError("Empty response from LLM")
        return result.strip()
    except Exception as e:
        logger.error(f"Generation task failed for item {item.id}: {e}")
        # Re-raise the exception so asyncio.gather can catch it
        raise


async def generate_data_for_items(
    items: List[EvaluationItem],
    system_prompt: str,
    config: GenerationConfig,
    api_keys: Dict[str, str],
    progress_callback: Optional[Callable[[int, int], None]] = None,
    batch_size: int = 5
) -> List[EvaluationItem]:
    """
    Generate data for each item using the configured LLM and template.
    """
    generator_config = config.data_generator_llm
    if not generator_config.api_key:
        generator_config.api_key = api_keys.get(generator_config.provider)
        
    if not generator_config.api_key:
        raise ValueError(f"No API key found for provider: {generator_config.provider}")
    
    client = create_llm_client(generator_config.provider, generator_config.api_key)
    
    try:
        template = jinja2.Template(config.data_generation_template)
        template.render(item=EvaluationItem(input="test", expected_output="test").model_dump(), context="test")
    except jinja2.TemplateError as e:
        raise ValueError(f"Invalid Jinja2 template in 'data_generation_template': {str(e)}")
    
    completed = 0
    total = len(items)
    
    for i in range(0, total, batch_size):
        batch = items[i:i + batch_size]
        tasks = [
            _generate_for_single_item(item, system_prompt, template, client, generator_config)
            for item in batch
        ]
        
        # return_exceptions=True ensures that if one task fails, the others complete.
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for item, result in zip(batch, results):
            if isinstance(result, Exception):
                error_msg = f"[ERROR: Generation failed - {str(result)}]"
            else:
                error_msg = None

            if error_msg:
                if config.mode == GenerationMode.GENERATE_OUTPUTS:
                    item.output = error_msg
                else:
                    item.expected_output = error_msg
            else:
                if config.mode == GenerationMode.GENERATE_OUTPUTS:
                    item.output = result
                else:
                    item.expected_output = result
            
            completed += 1
            if progress_callback:
                progress_callback(completed, total)
    
    return items


async def generate_outputs(
    items: List[EvaluationItem],
    actor_config: Dict[str, Any],
    progress_callback: Optional[Callable[[int, int], None]] = None,
    batch_size: int = 5,
) -> List[EvaluationItem]:
    """
    Legacy function for backward compatibility. Wraps the new generation logic.
    """
    # FIX: Correctly construct dependencies for the new generation function.
    provider = actor_config["provider"]
    api_key = actor_config.get("api_key")
    api_keys = {provider: api_key} if api_key else {}

    generation_config = GenerationConfig(
        mode=GenerationMode.GENERATE_OUTPUTS,
        # This uses the new default template for output generation
        data_generator_llm=LLMConfig(
            provider=provider,
            model=actor_config["model"],
            temperature=actor_config.get("temperature", 0.7),
            max_tokens=actor_config.get("max_tokens", 1000),
        )
    )
    
    system_prompt = actor_config.get("system_prompt", "Generate an appropriate output for the given input and expected output reference.")
    
    return await generate_data_for_items(
        items=items,
        system_prompt=system_prompt,
        config=generation_config,
        api_keys=api_keys,
        progress_callback=progress_callback,
        batch_size=batch_size
    )

# The original generate_single_output and validate_generation_config can remain as they were.
# They are not part of the new core flow but might be used elsewhere.
async def generate_single_output(
    input_text: str,
    actor_config: Dict[str, Any],
) -> str:
    """
    Generate a single output for testing or one-off generation.
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
    messages = [{"role": "user", "content": input_text}]
    if system_prompt:
        messages.insert(0, {"role": "system", "content": system_prompt})
    try:
        output = await client.generate(messages, **gen_params)
        return output
    except Exception as e:
        logger.error(f"Error generating single output: {e}")
        raise

def validate_generation_config(config: Dict[str, Any]) -> bool:
    """
    Validate that a legacy generation configuration has all required fields.
    """
    required_fields = ["provider", "model"]
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required field in generation config: {field}")
    return True