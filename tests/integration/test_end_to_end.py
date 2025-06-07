from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))

from core.data_models import EvalRecord
from core.evaluation import run_evaluation


def test_run_evaluation():
    records = [EvalRecord(prompt="hi", expected="hi", output="hi")]
    result = run_evaluation(records)
    assert len(result.scores) == 1
