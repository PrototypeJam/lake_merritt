import asyncio
import json
import logging
from typing import Any, Dict

from core.data_models import EvaluationItem, ScorerResult
from core.scoring.base import BaseScorer
from services.llm_clients import create_llm_client

log = logging.getLogger(__name__)

_PROMPT_TMPL = """You are an expert evaluator.
Analyse whether the FINAL SELECTED CRITERIA are the best subset of the CANDIDATE CRITERIA for achieving the USER GOAL, given the SEARCH SUMMARY.

USER GOAL:
{goal}

SEARCH SUMMARY:
{search}

CANDIDATE CRITERIA:
{generated}

FINAL SELECTED CRITERIA:
{selected}

Return JSON:
{{
  "score": 0-1,
  "passed": true/false,  // passed if score >= {threshold}
  "reasoning": "brief explanation"
}}
"""


class CriteriaSelectionJudgeScorer(BaseScorer):
    """LLM-as-Judge scoring of criteria-selection quality."""
    
    requires_api_key = True  # Criteria Selection Judge requires API key for LLM access

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.provider = self.config.get("provider", "openai")
        self.model = self.config.get("model", "gpt-4o-mini")
        self.temperature = self.config.get("temperature", 0.3)
        self.threshold = self.config.get("threshold", 0.7)
        self.api_key = self.config.get("api_key")
        self.client = None  # lazy

    # --- metadata for UI ---
    @property
    def name(self):
        return "criteria_selection_judge"

    @property
    def display_name(self):
        return "Criteria Selection Judge"

    @property
    def description(self):
        return "Uses an LLM to judge whether selected success-criteria are appropriate."

    # --- core logic ---
    async def score(self, item: EvaluationItem) -> ScorerResult:
        if self.client is None:
            self.client = create_llm_client(self.provider, self.api_key)

        md = item.metadata

        # Handle different criteria formats
        generated_criteria = md.get("generated_criteria", [])
        selected_criteria = md.get("selected_criteria", [])

        # Format generated criteria
        if generated_criteria and isinstance(generated_criteria[0], dict):
            generated_text = "\n".join(
                f"- {c.get('criteria', str(c))}" for c in generated_criteria
            )
        else:
            generated_text = "\n".join(f"- {c}" for c in generated_criteria)

        # Format selected criteria
        if selected_criteria and isinstance(selected_criteria[0], dict):
            selected_text = "\n".join(
                f"- {c.get('criteria', str(c))}" for c in selected_criteria
            )
        else:
            selected_text = "\n".join(f"- {c}" for c in selected_criteria)

        prompt = _PROMPT_TMPL.format(
            goal=md.get("user_goal", "No goal specified"),
            search=md.get("search_summary", "No search summary available"),
            generated=generated_text or "No criteria generated",
            selected=selected_text or "No criteria selected",
            threshold=self.threshold,
        )

        # Log prompt length for debugging
        log.info(f"Prompt length: {len(prompt)} characters")
        if len(prompt) > 3000:
            log.warning("Prompt may be too long, truncating search summary")
            # Truncate search summary if too long
            search_summary = (
                md.get("search_summary", "")[:500] + "..."
                if len(md.get("search_summary", "")) > 500
                else md.get("search_summary", "")
            )
            prompt = _PROMPT_TMPL.format(
                goal=md.get("user_goal", "No goal specified"),
                search=search_summary,
                generated=generated_text or "No criteria generated",
                selected=selected_text or "No criteria selected",
                threshold=self.threshold,
            )

        messages = [
            {"role": "system", "content": "You are a strict, unbiased evaluator."},
            {"role": "user", "content": prompt},
        ]

        try:
            raw = await self.client.generate(
                messages,
                model=self.model,
                temperature=self.temperature,
                max_tokens=600,
            )

            # Check if response is empty
            if not raw or raw.strip() == "":
                log.error("Empty response from LLM")
                return ScorerResult(
                    scorer_name=self.name,
                    score=0.0,
                    passed=False,
                    error="Empty response from LLM",
                    reasoning="The LLM returned an empty response.",
                )

            # Try to parse JSON
            try:
                data = json.loads(raw)
            except json.JSONDecodeError as je:
                log.error("Invalid JSON response: %s. Raw response: %s", je, raw[:200])
                return ScorerResult(
                    scorer_name=self.name,
                    score=0.0,
                    passed=False,
                    error=f"Invalid JSON: {str(je)}",
                    reasoning="LLM response was not valid JSON.",
                    details=(
                        {"raw_response": raw[:500]}
                        if self.config.get("debug")
                        else None
                    ),
                )

            score = float(data.get("score", 0.0))
            passed = bool(data.get("passed", score >= self.threshold))
            return ScorerResult(
                scorer_name=self.name,
                score=score,
                passed=passed,
                reasoning=data.get("reasoning", ""),
                details={"raw_response": raw} if self.config.get("debug") else None,
            )
        except Exception as e:
            log.error("Judge failed with unexpected error: %s", e)
            return ScorerResult(
                scorer_name=self.name,
                score=0.0,
                passed=False,
                error=str(e),
                reasoning="LLM call failed with unexpected error.",
            )
