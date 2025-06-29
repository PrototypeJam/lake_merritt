"""
core.ingestion.python_ingester
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A thin wrapper that lets an Eval Pack point to **any** Python script +
function which returns an iterable of `EvaluationItem`s.

Example usage in YAML
---------------------

ingestion:
  type: "python"
  script_path: "my_ingesters/otel_helper.py"
  entry_function: "ingest_otel_to_evalitems"
  config:
    trace_file: "manual_traces.json"
"""
from __future__ import annotations

import importlib
from pathlib import Path
from typing import Iterable, List, Any

from core.ingestion.base import BaseIngester, EvaluationItem


class PythonIngester(BaseIngester):
    """
    Dynamically imports `script_path`, fetches `entry_function`,
    and calls it with `self.config` (a `dict`).

    The target function **must** return an `Iterable[EvaluationItem]`.
    """

    def ingest(self) -> List[EvaluationItem]:
        script_path: str = self.config["script_path"]
        entry_func: str = self.config.get("entry_function", "ingest")

        # Make path import-friendly:  "folder/helper.py" -> "folder.helper"
        module_name = Path(script_path).with_suffix("").as_posix().replace("/", ".")
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            raise ImportError(
                f"PythonIngester could not import module '{module_name}'. "
                f"Check 'script_path' in your Eval Pack."
            ) from exc

        if not hasattr(module, entry_func):
            raise AttributeError(
                f"Module '{module_name}' has no attribute '{entry_func}'. "
                "Verify 'entry_function' in your Eval Pack."
            )

        fn = getattr(module, entry_func)
        items: Iterable[Any] = fn(self.config)  # type: ignore[arg-type]

        # Defensive: validate each yielded object
        eval_items: List[EvaluationItem] = []
        for idx, obj in enumerate(items):
            if not isinstance(obj, EvaluationItem):
                raise TypeError(
                    f"Function '{entry_func}' returned a non-EvaluationItem at "
                    f"index {idx}: {type(obj)}"
                )
            eval_items.append(obj)
        return eval_items