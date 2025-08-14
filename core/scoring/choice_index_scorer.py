# core/scoring/choice_index_scorer.py
from __future__ import annotations
import re
from typing import Any, Dict, Optional
from rapidfuzz import fuzz
from core.scoring.base import BaseScorer
from core.data_models import ScorerResult, EvaluationItem

class ChoiceIndexScorer(BaseScorer):
    """Maps model text to an index (0, 1, 2) by parsing letter/index or fuzzy matching."""

    @property
    def name(self) -> str:
        return "Choice Index"

    # Robust regex that catches "Answer: B", "final answer C", "Option-1", or bare "A/0"
    _ANS_RE = re.compile(r'(?i)\b(?:answer|final\s*answer|option|choice)?\s*[:\-]?\s*([ABC]|[012])\b')

    def _parse_letter_or_index(self, text: str) -> Optional[int]:
        m = self._ANS_RE.search((text or "").strip())
        if not m:
            return None
        token = m.group(1).lower()
        return {'a':0, 'b':1, 'c':2, '0':0, '1':1, '2':2}.get(token)

    def score(self, item: EvaluationItem, stage_config: Dict[str, Any]) -> ScorerResult:
        out = (item.output or "").strip()
        md = item.metadata or {}
        choices = [
            (md.get("choice_a") or md.get("ans0") or "").strip().lower(),
            (md.get("choice_b") or md.get("ans1") or "").strip().lower(),
            (md.get("choice_c") or md.get("ans2") or "").strip().lower()
        ]
        thresh = int(stage_config.get("fuzzy_match_threshold", 90))

        pred_idx = self._parse_letter_or_index(out)
        best_r = 100.0 if pred_idx is not None else -1

        if pred_idx is None:
            best_i = -1
            for i, choice_text in enumerate(choices):
                if not choice_text:
                    continue
                r = fuzz.token_sort_ratio(out.lower(), choice_text)
                if r > best_r:
                    best_i, best_r = i, r
            if best_r >= thresh:
                pred_idx = best_i

        gold_idx = md.get("correct_label_index")
        passed = False
        if pred_idx is not None and gold_idx is not None:
            passed = (int(pred_idx) == int(gold_idx))

        return ScorerResult(
            scorer_name="choice_index",
            score=1.0 if passed else 0.0,
            passed=passed,
            details={"predicted_label_index": pred_idx, "match_ratio": float(best_r)}
        )