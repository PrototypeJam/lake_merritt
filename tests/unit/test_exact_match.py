from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))

from core.data_models import EvalRecord
from core.scoring import exact_match


def test_exact_match_scores_correctly():
    rec = EvalRecord(prompt="", expected="hello", output="hello")
    score = exact_match.score(rec, {})
    assert score.value == 1.0
