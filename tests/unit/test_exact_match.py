from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))

from core.data_models import EvaluationItem
from core.scoring import ExactMatchScorer


def test_exact_match_scores_correctly():
    item = EvaluationItem(id=1, input="", expected_output="hello", output="hello")
    scorer = ExactMatchScorer()
    result = scorer.score(item)
    assert result.score == 1.0
    assert result.passed
