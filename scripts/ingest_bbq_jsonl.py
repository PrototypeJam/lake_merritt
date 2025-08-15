# scripts/ingest_bbq_jsonl.py
from __future__ import annotations
import csv, json
from pathlib import Path
from typing import Dict, Any, Generator
from core.data_models import EvaluationItem

def _load_additional_metadata(root: Path) -> Dict[tuple, Dict[str, Any]]:
    """Load the BBQ supplemental metadata CSV."""
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

def _resolve_dataset_root(path_str: str) -> Path:
    """
    Resolve a user-provided path (absolute or repo-relative) to a local directory
    that contains:
        data/*.jsonl
        supplemental/additional_metadata.csv

    Robust across:
      - Streamlit Community Cloud (Linux, repo root as CWD)
      - Local dev on macOS/Linux/Windows
      - Windows-style inputs with backslashes
    """
    raw = (path_str or "").strip()
    if not raw:
        raise ValueError("No path provided. The uploaded file must contain the dataset root path.")

    # Normalize path separators so Windows-style inputs work on Linux.
    # On POSIX, backslash is just a normal character, not a separator.
    raw = raw.replace("\\", "/")

    p = Path(raw)
    candidates = []

    # 1) As-is (works for absolute paths)
    candidates.append(p)

    # 2) Relative to current working directory (repo root on Streamlit Cloud)
    candidates.append(Path.cwd() / p)

    # 3) Relative to repository root inferred from this file location:
    # scripts/ingest_bbq_jsonl.py -> repo root = parent of 'scripts'
    repo_root = Path(__file__).resolve().parent.parent
    candidates.append(repo_root / p)

    for c in candidates:
        if c.is_dir():
            data_dir = c / "data"
            supp_file = c / "supplemental" / "additional_metadata.csv"
            if data_dir.is_dir() and any(data_dir.glob("*.jsonl")) and supp_file.is_file():
                return c

    tried = "\n".join(str(x) for x in candidates)
    raise FileNotFoundError(
        "Could not resolve the dataset directory from the provided path.\n"
        "Expected to find 'data/*.jsonl' and 'supplemental/additional_metadata.csv'.\n"
        f"Tried:\n{tried}\n"
        "Tips:\n"
        "  • Use a repository-relative path like 'data/BBQ-mini'.\n"
        "  • Ensure directory names are exact-case matches (Linux is case-sensitive).\n"
        "  • Avoid Windows backslashes; if present, they are normalized automatically."
    )

def ingest(config: Dict[str, Any]) -> Generator[EvaluationItem, None, None]:
    """
    Entry point for PythonIngester: accepts a config dict.
    Expects the uploaded file handle at config['path_file'] (or 'data_file' / 'trace_file').
    The file contains a single path (absolute or repo-relative) to the BBQ dataset root.
    """
    file_obj = config.get("path_file") or config.get("data_file") or config.get("trace_file")
    if not file_obj:
        raise ValueError("No path file provided in config. Expected 'path_file', 'data_file', or 'trace_file'.")

    data_root_str = _read_text(file_obj)
    root = _resolve_dataset_root(data_root_str)

    # Load supplemental metadata
    meta = _load_additional_metadata(root)

    # Process all JSONL files in the data directory
    data_dir = root / "data"
    if not data_dir.exists():
        raise FileNotFoundError(f"BBQ data directory not found at {data_dir}. Expected: <data_root>/data/")

    for jf in sorted(data_dir.glob("*.jsonl")):
        category = jf.stem
        with jf.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue

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
