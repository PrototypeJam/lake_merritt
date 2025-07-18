"""
core.ingestion.python_ingester
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A thin wrapper that lets an Eval Pack point to **any** Python script +
function which returns an iterable of `EvaluationItem`s.
"""
from __future__ import annotations

import importlib
from pathlib import Path
from typing import Iterable, List, Any, Dict

from core.ingestion.base import BaseIngester, EvaluationItem


class PythonIngester(BaseIngester):
    """
    Dynamically imports `script_path`, fetches `entry_function`,
    and calls it with a configuration dictionary.

    The target function **must** return an `Iterable[EvaluationItem]`.
    """

    # FIX 1: Corrected the method signature to accept 'data' and 'config'
    def ingest(self, data: Any, config: Dict[str, Any]) -> List[EvaluationItem]:
        """
        Dynamically loads and executes an entry function from a specified script.

        Args:
            data: The raw data passed from the evaluation engine (e.g., an uploaded file object).
            config: The configuration dictionary from the Eval Pack's 'ingestion' block.
        
        Returns:
            A list of EvaluationItem objects.
        """
        # FIX 2: Use the 'config' dictionary that is passed in, not 'self.config'
        script_path: str = config["script_path"]
        entry_func: str = config.get("entry_function", "ingest")

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

        # FIX 3: Prepare the configuration for the external function.
        # The uploaded file object (`data`) is the value our external script
        # expects for its 'trace_file' parameter. We inject it into the config.
        external_config = config.copy()
        external_config["trace_file"] = data

        fn = getattr(module, entry_func)
        
        # Call the external function with the prepared configuration
        items: Iterable[Any] = fn(external_config)

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