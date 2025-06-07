"""
LLM-as-a-Judge scorer - uses an LLM to evaluate outputs.
"""
import json
import asyncio
import logging
from typing import Dict, Any, Optional

from core.scoring.base import BaseScorer
from core.data_models import EvaluationItem, ScorerResult
from services.llm_clients import create_llm_client

logger = logging.getLogger(__name__)


class LLMJudgeScorer(BaseScorer):
    """Scorer that uses an LLM to judge the quality of outputs."""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.provider = self.config.get("provider", "openai")
        self.model = self.config.get("model", "gpt-4")
        self.temperature = self.config.get("temperature", 0.3)
        self.max_tokens = self.config.get("max_tokens", 1000)
        self.system_prompt = self.config.get("system_prompt", self._default_system_prompt())
        self.api_key = self.config.get("api_key")
        self.threshold = self.config.get("threshold", 0.7)

        # LLM client is lazily created on first use
        self.client = None

    @property
    def name(self) -> str:
        return "LLM Judge"

    @property
    def description(self) -> str:
        return f"Uses {self.model} to evaluate output quality"

    def _default_system_prompt(self) -> str:
        """Get the default system prompt for the judge."""
        return """You are an expert evaluator. Compare the actual output to the expected output and provide:
1. A score from 0.0 to 1.0 (where 1.0 is perfect match/quality)
2. A brief reasoning for your score
3. Any specific errors or discrepancies noted

Consider the following criteria:
- Factual accuracy
- Completeness of the answer
- Clarity and coherence
- Relevance to the input

Respond in JSON format:
{
    "score": 0.0-1.0,
    "reasoning": "explanation",
    "errors": ["error1", "error2"] or []
}"""

    async def score(self, item: EvaluationItem) -> ScorerResult:
        """Score an item using LLM judgment."""
        if self.client is None:
            self.client = create_llm_client(self.provider, self.api_key)

        if item.output is None:
            return ScorerResult(
                scorer_name="llm_judge",
                score=0.0,
                passed=False,
                reasoning="No output provided",
            )

        # Construct the evaluation prompt
        user_prompt = f"""Please evaluate the following:

Input/Question: {item.input}

Expected Output: {item.expected_output}

Actual Output: {item.output}

Provide your evaluation in the specified JSON format."""

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            # Call the LLM
            response = await self.client.generate(
                messages,
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            # Parse the response
            try:
                # Try to extract JSON from the response
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = response[json_start:json_end]
                    result = json.loads(json_str)
                else:
                    raise ValueError("No JSON found in response")

            except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"Failed to parse LLM response as JSON: {e}")
                logger.debug(f"Raw response: {response}")

                # Fallback: try to extract score from text
                score = self._extract_score_from_text(response)
                result = {
                    "score": score,
                    "reasoning": response,
                    "errors": ["Failed to parse structured response"]
                }

            # Extract values with defaults
            score = float(result.get("score", 0.0))
            score = max(0.0, min(1.0, score))
            reasoning = result.get("reasoning", "No reasoning provided")
            errors = result.get("errors", [])

            passed = score >= self.threshold

            return ScorerResult(
                scorer_name="llm_judge",
                score=score,
                passed=passed,
                reasoning=reasoning,
                details={
                    "model": self.model,
                    "threshold": self.threshold,
                    "errors": errors,
                    "raw_response": response if self.config.get("include_raw_response", False) else None,
                }
            )

        except Exception as e:
            logger.error(f"Error in LLM judge scoring: {e}")
            return ScorerResult(
                scorer_name="llm_judge",
                score=0.0,
                passed=False,
                error=str(e),
                reasoning=f"Failed to get LLM judgment: {str(e)}",
            )

    def _extract_score_from_text(self, text: str) -> float:
        """Try to extract a numeric score from text response."""
        import re

        patterns = [
            r'score[:\s]+([0-9]*\.?[0-9]+)',
            r'([0-9]*\.?[0-9]+)/10',
            r'([0-9]+)%',
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                value = float(match.group(1))
                if '%' in pattern:
                    return value / 100.0
                elif '/10' in pattern:
                    return value / 10.0
                elif value <= 1.0:
                    return value
                elif value <= 10:
                    return value / 10.0
                elif value <= 100:
                    return value / 100.0

        return 0.0


class StructuredLLMJudgeScorer(LLMJudgeScorer):
    """Enhanced LLM Judge that enforces structured output."""

    @property
    def name(self) -> str:
        return "Structured LLM Judge"

    @property
    def description(self) -> str:
        return f"Uses {self.model} with structured output for consistent evaluation"

    async def score(self, item: EvaluationItem) -> ScorerResult:
        """Score using structured output capabilities."""
        if self.client is None:
            self.client = create_llm_client(self.provider, self.api_key)

        if item.output is None:
            return ScorerResult(
                scorer_name="structured_llm_judge",
                score=0.0,
                passed=False,
                reasoning="No output provided",
            )

        evaluation_schema = {
            "type": "object",
            "properties": {
                "score": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "description": "Score from 0.0 to 1.0"
                },
                "reasoning": {
                    "type": "string",
                    "description": "Explanation for the score"
                },
                "errors": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of specific errors found"
                },
                "suggestions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Suggestions for improvement"
                }
            },
            "required": ["score", "reasoning"]
        }

        user_prompt = f"""Evaluate the actual output against the expected output.

Input: {item.input}
Expected Output: {item.expected_output}
Actual Output: {item.output}"""

        try:
            if self.provider == "openai" and hasattr(self.client, 'generate_structured'):
                result = await self.client.generate_structured(
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    model=self.model,
                    schema=evaluation_schema,
                    temperature=self.temperature,
                )
            else:
                return await super().score(item)

            score = float(result.get("score", 0.0))
            passed = score >= self.threshold

            return ScorerResult(
                scorer_name="structured_llm_judge",
                score=score,
                passed=passed,
                reasoning=result.get("reasoning", ""),
                details={
                    "model": self.model,
                    "threshold": self.threshold,
                    "errors": result.get("errors", []),
                    "suggestions": result.get("suggestions", []),
                }
            )

        except Exception as e:
            logger.error(f"Error in structured LLM judge: {e}")
            return await super().score(item)