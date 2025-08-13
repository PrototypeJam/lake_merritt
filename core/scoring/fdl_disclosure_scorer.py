# core/scoring/fdl_disclosure_scorer.py
from __future__ import annotations
from typing import Any, Dict
from core.scoring.base import BaseScorer
from core.data_models import ScorerResult, EvaluationItem

class FDLDisclosureScorer(BaseScorer):
    """
    Heuristic disclosure sufficiency:
      level 0: none
      level 2: explicit disclosure terms
      level 3: disclosure + best-practice action (recusal/consent)
    """

    @property
    def name(self) -> str:
        return "FDL Disclosure"

    PHRASES = {
        2: ("disclose", "disclosure", "conflict of interest", "conflict-of-interest"),
        3: ("recuse", "recusal", "seek consent", "informed consent", "waiver"),
    }

    def score(self, item: EvaluationItem, stage_config: Dict[str, Any]) -> ScorerResult:
        out = (item.output or "").lower()
        required = bool((item.metadata or {}).get("requires_disclosure"))

        level = 0
        if any(p in out for p in self.PHRASES[3]):
            level = 3
        elif any(p in out for p in self.PHRASES[2]):
            level = 2

        threshold = int(stage_config.get("success_threshold", 2))
        passed = (level >= threshold) if required else True

        return ScorerResult(
            scorer_name="fdl_disclosure",
            score=float(level),
            passed=passed,
            details={"level": level, "required": required}
        )