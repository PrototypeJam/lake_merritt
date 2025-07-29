"""
LLM-as-a-Judge scorers: standard and structured output variants.

This module defines two main scorers for the Lake Merritt evaluation framework:
- LLMJudgeScorer: General-purpose scorer for LLM-as-a-judge, relying on YAML prompt templates, expecting JSON output.
- StructuredLLMJudgeScorer: Strict schema-based scorer, enforcing that the LLM returns output conforming to a specified schema, using function calling or OpenAI's JSON mode if supported.

Key architectural principles (per latest design/your prior requests):
- ALL configuration is passed at scoring time (via the stage_config argument), NOT at construction, to support fully stateless scorers and per-stage prompts/models/settings.
- Client caching is only used to avoid re-instantiating LLM clients for the same provider/API key.
- Prompt rendering is done via Jinja2, using the fields of EvaluationItem, for full flexibility.
- Error handling is robust: template errors, LLM errors, and JSON parsing errors all result in a 0.0 score and a detailed error field.

Extensive comments are provided throughout for future maintainers/contributors.

Copyright (c) 2024 Lake Merritt, MIT, OpenAI, contributors.
"""

import asyncio
import json
import logging
from typing import Any, Dict, Optional

import jinja2
from core.data_models import EvaluationItem, ScorerResult
from core.scoring.base import BaseScorer
from services.llm_clients import create_llm_client

logger = logging.getLogger(__name__)

class LLMJudgeScorer(BaseScorer):
    """
    Standard LLM-as-a-judge scorer.
    - Prompts are defined in your Eval Pack YAML and rendered via Jinja2.
    - Expects LLM output in JSON with at least "score" (float 0-1) and "reasoning" (str).
    - All config is passed via stage_config, NOT at construction.
    - Caches LLM clients by (provider, api_key).
    """
    requires_api_key = True

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        # Cache LLM client instances for reuse. The key is (provider, api_key).
        self.client_cache = {}

    @property
    def name(self) -> str:
        return "LLM Judge"

    @property
    def description(self) -> str:
        return "Uses an LLM to judge output quality based on flexible YAML/Jinja2 prompts."

    def _get_client(self, provider: str, api_key: Optional[str]):
        """
        Retrieve or lazily create an LLM client for (provider, api_key).
        This allows you to call OpenAI, Anthropic, Gemini, etc., in a single pipeline.
        """
        client_key = (provider, api_key)
        if client_key not in self.client_cache:
            self.client_cache[client_key] = create_llm_client(provider, api_key)
        return self.client_cache[client_key]

    def _default_system_prompt(self) -> str:
        """
        Provides a safe default system prompt if none is set in stage_config.
        """
        return (
            "You are an expert evaluator. "
            "Respond ONLY in valid JSON with \"score\" (0.0-1.0) and \"reasoning\" fields."
        )
    
    def _default_user_prompt_template(self) -> str:
        """Provides a default user prompt for backward compatibility."""
        return """
Compare the actual output to the expected output for the given input.

Input: {{ input }}
Expected Output: {{ expected_output }}
Actual Output: {{ output }}

Respond in JSON format with:
- "score": 0.0 to 1.0
- "reasoning": explanation of your evaluation
""".strip()

    async def score(self, item: EvaluationItem, stage_config: Dict[str, Any]) -> ScorerResult:
        """
        The main scoring method. Follows the Lake Merritt stateless scorer contract:
        - Accepts per-stage config at runtime.
        - Renders the prompt using Jinja2.
        - Sends prompt(s) to the LLM, requesting JSON output.
        - Parses and returns score, reasoning, and any other returned fields.
        """
        provider = stage_config.get("provider", "openai")
        api_key = stage_config.get("api_key")
        model = stage_config.get("model", "gpt-4o")
        temperature = stage_config.get("temperature", 0.3)
        max_tokens = stage_config.get("max_tokens", 1000)
        threshold = stage_config.get("threshold", 0.7)
        system_prompt = stage_config.get("system_prompt", self._default_system_prompt())
        user_prompt_template = stage_config.get("user_prompt_template") or self._default_user_prompt_template()

        if not user_prompt_template or not user_prompt_template.strip():
            raise ValueError(
                "'user_prompt_template' is a required key in the 'config' block for the LLM Judge scorer. "
                "Please define it in the scorer configuration UI or your Eval Pack."
            )

        client = self._get_client(provider, api_key)

        if item.output is None:
            return ScorerResult(
                scorer_name=self.name,
                score=0.0,
                passed=False,
                reasoning="No output provided"
            )

        # Render the user prompt using Jinja2.
        try:
            template = jinja2.Template(user_prompt_template)
            # FIX: Unpack the item's dictionary into keyword arguments.
            # This makes `input`, `output`, `expected_output`, etc., available as
            # top-level variables in the Jinja2 template.
            user_prompt = template.render(**item.model_dump())
        except Exception as e:
            logger.error(f"Jinja2 template error for item '{item.id}': {e}", exc_info=True)
            return ScorerResult(
                scorer_name=self.name,
                score=0.0,
                passed=False,
                error=f"Jinja2 template error: {e}"
            )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            # Call LLM API (client should wrap OpenAI/Anthropic/Gemini/etc.).
            response = await client.generate(
                messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            # Attempt to extract first valid JSON object from response
            try:
                json_start = response.find("{")
                json_end = response.rfind("}") + 1
                if json_start >= 0 and json_end > json_start:
                    result = json.loads(response[json_start:json_end])
                else:
                    raise ValueError("No JSON object found in LLM response")
            except (json.JSONDecodeError, ValueError) as e:
                logger.error(
                    f"Failed to parse LLM response as JSON: {e}. Raw response: {response}"
                )
                return ScorerResult(
                    scorer_name=self.name,
                    score=0.0,
                    passed=False,
                    error=f"Invalid JSON response: {e}",
                    details={"raw_response": response}
                )

            # Defensive: Ensure 'score' is float and within [0,1]
            score = result.get("score", 0.0)
            try:
                score = float(score)
            except (ValueError, TypeError):
                score = 0.0
            score = max(0.0, min(1.0, score))

            passed = score >= threshold
            reasoning = result.get("reasoning", "No reasoning provided.")

            details = {"model": model, "threshold": threshold}
            details.update(result)

            return ScorerResult(
                scorer_name=self.name,
                score=score,
                passed=passed,
                reasoning=reasoning,
                details=details,
            )

        except Exception as exc:
            logger.error(f"LLM judge API call failed: {exc}", exc_info=True)
            return ScorerResult(
                scorer_name=self.name,
                score=0.0,
                passed=False,
                error=str(exc)
            )


class StructuredLLMJudgeScorer(LLMJudgeScorer):
    """
    Structured LLM Judge scorer: Enforces strict JSON schema output from the LLM.

    - Uses OpenAI function calling/JSON mode, Anthropic tool use, or Gemini function calling if supported.
    - Falls back to free-form prompt-based judging (parent class) if structured call is not available for the provider.
    - Useful for business-critical/regulated cases, for test suites, or for advanced scoring (returning errors, suggestions, multi-field output).

    Your LLM client class must implement a .generate_structured(...) method (like OpenAI's 'functions' API),
    or this will fall back to the normal .generate(...) method.

    Configuration is always per-stage, as above.
    """
    requires_api_key = True

    @property
    def name(self) -> str:
        return "Structured LLM Judge"

    @property
    def description(self) -> str:
        return (
            "Uses an LLM with enforced structured output (JSON schema) for evaluation, "
            "including optional error/suggestions fields. "
            "Falls back to regular prompt-based LLM judge if not supported."
        )

    async def score(self, item: EvaluationItem, stage_config: Dict[str, Any]) -> ScorerResult:
        """
        Main scoring entry point for structured mode.
        - If your LLM client supports generate_structured(...), uses it with the provided schema.
        - Otherwise, falls back to the standard prompt-based judge.
        - All config is per-stage, never at construction.
        """

        provider = stage_config.get("provider", "openai")
        api_key = stage_config.get("api_key")
        model = stage_config.get("model", "gpt-4o")
        temperature = stage_config.get("temperature", 0.3)
        max_tokens = stage_config.get("max_tokens", 1000)
        threshold = stage_config.get("threshold", 0.7)
        system_prompt = stage_config.get("system_prompt", self._default_system_prompt())
        user_prompt_template = stage_config.get("user_prompt_template")

        client = self._get_client(provider, api_key)

        if item.output is None:
            return ScorerResult(
                scorer_name=self.name,
                score=0.0,
                passed=False,
                reasoning="No output provided"
            )

        # Define the JSON schema expected from the LLM
        evaluation_schema = {
            "type": "object",
            "properties": {
                "score": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "description": "Score from 0.0 to 1.0",
                },
                "reasoning": {
                    "type": "string",
                    "description": "Explanation for the score",
                },
                "errors": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of specific errors found",
                },
                "suggestions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Suggestions for improvement",
                },
            },
            "required": ["score", "reasoning"],
        }

        # Render the user prompt using Jinja2 (as with standard LLMJudge)
        try:
            template = jinja2.Template(user_prompt_template)
            # FIX: Unpack here as well for consistency.
            user_prompt = template.render(**item.model_dump())
        except Exception as e:
            logger.error(f"Jinja2 template error for item '{item.id}': {e}", exc_info=True)
            return ScorerResult(
                scorer_name=self.name,
                score=0.0,
                passed=False,
                error=f"Jinja2 template error: {e}"
            )

        # Try to call the LLM in structured mode (if available)
        try:
            # Only attempt structured mode if the client supports it
            if hasattr(client, "generate_structured"):
                response = await client.generate_structured(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    model=model,
                    schema=evaluation_schema,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                # If OpenAI or Anthropic, the response should already be parsed as a dict
                result = response if isinstance(response, dict) else json.loads(response)
            else:
                # If not supported, fallback to prompt-based judging
                logger.info("Structured mode not supported for this LLM client/provider. Falling back to prompt-based LLM judge.")
                return await super().score(item, stage_config)

            # Defensive: Ensure 'score' is a float and within [0,1]
            score = result.get("score", 0.0)
            try:
                score = float(score)
            except (ValueError, TypeError):
                score = 0.0
            score = max(0.0, min(1.0, score))

            passed = score >= threshold
            reasoning = result.get("reasoning", "No reasoning provided.")

            details = {"model": model, "threshold": threshold}
            details.update(result)

            return ScorerResult(
                scorer_name=self.name,
                score=score,
                passed=passed,
                reasoning=reasoning,
                details=details,
            )

        except Exception as e:
            logger.error(f"Error in structured LLM judge: {e}", exc_info=True)
            # Fallback to base judge on error
            return await super().score(item, stage_config)

# End of file: core/scoring/llm_judge.py