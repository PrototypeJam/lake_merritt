# scripts/ingest_bbq_jsonl.py
from __future__ import annotations
import csv, json
from pathlib import Path
from typing import Dict, Any, Generator
from core.data_models import EvaluationItem

def _load_additional_metadata(root: Path) -> Dict[tuple, Dict[str, Any]]:
    md: Dict[tuple, Dict[str, Any]] = {}
    path = root / "supplemental" / "additional_metadata.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Required BBQ metadata not found at {path}. "
            f"Expected: <data_root>/supplemental/additional_metadata.csv"
        )
    with path.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            key = (row["category"], row["example_id"])
            md[key] = row
    return md

def _read_text(obj: Any) -> str:
    """Return UTF-8 text from a path, file-like, bytes, or str."""
    if obj is None:
        return ""
    if hasattr(obj, "read"):
        data = obj.read()
        if isinstance(data, bytes):
            return data.decode("utf-8")
        return str(data)
    if isinstance(obj, (bytes, bytearray)):
        return obj.decode("utf-8")
    return str(obj)

def ingest(config: Dict[str, Any]) -> Generator[EvaluationItem, None, None]:
    """
    Entry point for PythonIngester: accepts a config dict.
    Expects the uploaded file handle at config['trace_file'] (or 'data_file'/'path_file').
    The file contains a single absolute path line to the BBQ dataset root.
    """
    file_obj = config.get("trace_file") or config.get("data_file") or config.get("path_file")
    data_root_str = _read_text(file_obj).strip()

    if not data_root_str:
        raise ValueError("No path provided in the uploaded file. Ensure bbq_path.txt contains the dataset root path.")

    root = Path(data_root_str)
    if not root.is_dir():
        raise FileNotFoundError(f"The provided path '{root}' is not a valid directory.")

    # Handle downloads that unzip into BBQ_full/
    if not (root / "data").exists() and (root / "BBQ_full" / "data").exists():
        root = root / "BBQ_full"

    meta = _load_additional_metadata(root)
    data_dir = root / "data"
    if not data_dir.exists():
        raise FileNotFoundError(f"BBQ data directory not found at {data_dir}. Expected: <data_root>/data/")

    for jf in data_dir.glob("*.jsonl"):
        category = jf.stem
        with jf.open("r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                ex_id = str(rec.get("example_id"))
                add = meta.get((category, ex_id), {})

                # Capture ans*_text and ans*_info to enable official 'unknown' filtering
                ans0_text = rec.get("ans0_text") or rec.get("ans0") or ""
                ans1_text = rec.get("ans1_text") or rec.get("ans1") or ""
                ans2_text = rec.get("ans2_text") or rec.get("ans2") or ""
                ans0_info = rec.get("ans0_info") or ""
                ans1_info = rec.get("ans1_info") or ""
                ans2_info = rec.get("ans2_info") or ""

                yield EvaluationItem(
                    id=f"bbq_{category}_{ex_id}",
                    input=(rec.get("context", "") or "") + "\n\n" + (rec.get("question", "") or ""),
                    output=None,
                    expected_output=rec.get(f"ans{rec.get('label')}", ""),
                    metadata={
                        "category": category,
                        "example_id": ex_id,
                        "ans0": rec.get("ans0", ""),
                        "ans1": rec.get("ans1", ""),
                        "ans2": rec.get("ans2", ""),
                        "ans0_text": ans0_text,
                        "ans1_text": ans1_text,
                        "ans2_text": ans2_text,
                        "ans0_info": ans0_info,
                        "ans1_info": ans1_info,
                        "ans2_info": ans2_info,
                        "correct_label_index": rec.get("label"),
                        "context_condition": rec.get("context_condition"),
                        "question_polarity": rec.get("question_polarity"),
                        "target_loc": int(add.get("target_loc")) if add.get("target_loc") else None,
                    },
                )