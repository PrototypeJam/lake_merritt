"""
LLM-as-a-Judge scorer - uses an LLM to evaluate outputs.
"""

import asyncio
import json
import logging
from typing import Any, Dict, Optional

from core.data_models import EvaluationItem, ScorerResult
from core.scoring.base import BaseScorer
from services.llm_clients import create_llm_client

logger = logging.getLogger(__name__)


class LLMJudgeScorer(BaseScorer):
    """Scorer that uses an LLM to judge the quality of outputs."""
    
    requires_api_key = True  # LLM Judge requires API key for model access

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.provider = self.config.get("provider", "openai")
        self.model = self.config.get("model", "gpt-4")
        self.temperature = self.config.get("temperature", 0.3)
        self.max_tokens = self.config.get("max_tokens", 1000)
        self.system_prompt = self.config.get(
            "system_prompt", self._default_system_prompt()
        )
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
1. A boolean 'passed' field (true if the output meets acceptable standards, false otherwise)
2. A score from 0.0 to 1.0 (where 1.0 is perfect match/quality)
3. A brief reasoning for your score and pass/fail decision
4. Any specific errors or discrepancies noted

Consider the following criteria:
- Factual accuracy
- Completeness of the answer
- Clarity and coherence
- Relevance to the input

Respond in JSON format:
{
    "passed": true/false,
    "score": 0.0-1.0,
    "reasoning": "explanation",
    "errors": ["error1", "error2"] or []
}"""

    async def score(self, item: EvaluationItem) -> ScorerResult:
        """Score an item using LLM judgment (pack-aware)."""
        if self.client is None:
            self.client = create_llm_client(self.provider, self.api_key)

        if item.output is None:
            return ScorerResult(
                scorer_name="llm_judge",
                score=0.0,
                passed=False,
                reasoning="No output provided",
            )

        # ---------------- NEW: allow pack-supplied template ----------------
        user_prompt_template: str | None = self.config.get("user_prompt_template")
        if user_prompt_template:
            # Convert the entire item to a dictionary to make all its fields
            # (including a JSON string of the metadata) available for formatting.
            item_dict = item.model_dump(mode="json")
            user_prompt = user_prompt_template.format(**item_dict)
        else:
            user_prompt = (
                f"Please evaluate the following:\n\n"
                f"Input/Question: {item.input}\n\n"
                f"Expected Output: {item.expected_output}\n\n"
                f"Actual Output: {item.output}\n\n"
                "Provide your evaluation in the specified JSON format."
            )
        # -------------------------------------------------------------------

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            response = await self.client.generate(
                messages,
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            # ---------- attempt to extract JSON block ----------
            try:
                json_start = response.find("{")
                json_end = response.rfind("}") + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = response[json_start:json_end]
                    result = json.loads(json_str)
                else:
                    raise ValueError("No JSON found in response")
            except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"Failed to parse LLM response as JSON: {e}")
                logger.debug(f"Raw response: {response}")
                score_from_text = self._extract_score_from_text(response)
                result = {
                    "score": score_from_text,
                    "reasoning": response,
                    "errors": ["Failed to parse structured response"],
                }
            # ----------------------------------------------------

            score = float(result.get("score", 0.0))
            score = max(0.0, min(1.0, score))  # clamp 0-1
            reasoning = result.get("reasoning", "No reasoning provided")
            errors = result.get("errors", [])

            if "passed" in result and isinstance(result["passed"], bool):
                passed = result["passed"]
            else:
                passed = score >= self.threshold
                # Warn only if using default prompt
                if not user_prompt_template:
                    logger.warning(
                        f"LLM response for item '{item.id}' missing 'passed'. "
                        f"Falling back to threshold check (score {score:.2f} >= {self.threshold})."
                    )

            return ScorerResult(
                scorer_name="llm_judge",
                score=score,
                passed=passed,
                reasoning=reasoning,
                details={
                    "model": self.model,
                    "threshold": self.threshold,
                    "errors": errors,
                    "raw_response": (
                        response if self.config.get("include_raw_response") else None
                    ),
                },
            )

        except Exception as exc:
            logger.error(f"Error in LLM judge scoring: {exc}")
            return ScorerResult(
                scorer_name="llm_judge",
                score=0.0,
                passed=False,
                error=str(exc),
                reasoning=f"Failed to obtain LLM judgment: {exc}",
            )

    def _extract_score_from_text(self, text: str) -> float:
        """Try to extract a numeric score from text response."""
        import re

        patterns = [
            r"score[:\s]+([0-9]*\.?[0-9]+)",
            r"score\s+is\s+(?:about\s+)?([0-9]*\.?[0-9]+)",
            r"([0-9]*\.?[0-9]+)\s*(?:out\s*of\s*)?/?\s*1(?:\.0)?",
            r"([0-9]*\.?[0-9]+)\s*/\s*10",
            r"([0-9]+)\s*out\s*of\s*10",
            r"([0-9]+)%",
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                value = float(match.group(1))
                if "%" in pattern:
                    return value / 100.0
                elif "/10" in pattern:
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
    
    requires_api_key = True  # Structured LLM Judge requires API key for model access

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

        user_prompt = f"""Evaluate the actual output against the expected output.

Input: {item.input}
Expected Output: {item.expected_output}
Actual Output: {item.output}"""

        try:
            if self.provider == "openai" and hasattr(
                self.client, "generate_structured"
            ):
                result = await self.client.generate_structured(
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": user_prompt},
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
                },
            )

        except Exception as e:
            logger.error(f"Error in structured LLM judge: {e}")
            return await super().score(item)
