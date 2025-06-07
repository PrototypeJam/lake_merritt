Below is a **concrete, high-level implementation plan**—directory tree first, then concise notes on the purpose and key contents of every file or sub-package.  This should give you everything you need to scaffold the repo and start filling in modules incrementally.

---

## 1 Directory / File Layout

```
ai-eval-workbench/
├─ README.md
├─ pyproject.toml          # Poetry (or hatch) – dep mgmt, tool config
├─ requirements.txt        # fallback for pip-install
├─ .env.template           # names of env vars (API keys etc.)
├─ streamlit_app.py        # thin launcher → pages/
│
├─ app/                    # Streamlit UI layer only
│  ├─ __init__.py
│  ├─ state.py             # helpers for st.session_state
│  └─ pages/
│     ├─ 1_⚙️_config.py
│     ├─ 2_📄_eval_setup.py
│     ├─ 3_📊_results.py
│     └─ 4_⬇️_downloads.py
│
├─ core/                   # pure Python “business logic”
│  ├─ __init__.py
│  ├─ data_models.py       # Pydantic objects for everything
│  ├─ ingestion.py         # CSV → List[EvalRecord]
│  ├─ generation.py        # model “actor” output creation
│  ├─ evaluation.py        # orchestration: run_scorers()
│  ├─ reporting.py         # summary stats, JSON/CSV writers
│  ├─ logging_config.py
│  └─ scoring/             # pluggable scorers live here
│      ├─ __init__.py
│      ├─ exact_match.py
│      ├─ fuzzy_match.py
│      └─ llm_judge.py
│
├─ services/               # external integrations
│  ├─ __init__.py
│  └─ llm_clients.py       # OpenAI, Anthropic, Gemini wrappers
│
├─ utils/
│  ├─ __init__.py
│  ├─ file_cache.py        # simple disk cache for rate-limit relief
│  └─ telemetry.py         # placeholder OpenTelemetry hooks
│
├─ tests/
│  ├─ unit/
│  │   └─ test_exact_match.py …
│  └─ integration/
│      └─ test_end_to_end.py
│
└─ .github/
    └─ workflows/
        └─ ci.yml          # lint, unit tests
```

*(Emoji prefixes in `pages/` keep Streamlit tabs ordered.)*

---

## 2 Module Responsibilities

| Path                               | Core Responsibility                                                                                                    | Implementation Notes                                          |
| ---------------------------------- | ---------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------- |
| `streamlit_app.py`                 | `streamlit run` entrypoint. Imports `app.pages.*`; holds nothing else.                                                 | Keeps CLI simple and unopinionated.                           |
| **`app/state.py`**                 | Typed wrappers around `st.session_state` (config, uploaded data, results).                                             | Avoids raw string keys scattered across pages.                |
| **`app/pages/1_⚙️_config.py`**     | Page 1 UI: API keys, default judge model params. Writes to `state`.                                                    | Validate keys immediately with ping-call (optional).          |
| **`app/pages/2_📄_eval_setup.py`** | Page 2 UI: Mode A vs B, file upload, scorer & actor selection, “Start Evaluation”.                                     | Delegates all heavy lifting to `core`.                        |
| **`app/pages/3_📊_results.py`**    | Reads `state.results`; shows KPI cards, `st.dataframe`, expandable JSON reasoning.                                     | Charts via `st.altair_chart` or Plotly later.                 |
| **`app/pages/4_⬇️_downloads.py`**  | Builds CSV/JSON bytes from `core.reporting`; exposes `st.download_button`.                                             | Future placeholders for logs/traces.                          |
| **`core/data_models.py`**          | Pydantic classes: `EvalRecord`, `Score`, `RunMetadata`, `RunResult`.                                                   | Single-source schema for I/O, scoring, reporting.             |
| **`core/ingestion.py`**            | Validates uploaded CSV, maps to `List[EvalRecord]`.                                                                    | Raises rich `pydantic.ValidationError` for UI display.        |
| **`core/generation.py`**           | For Mode B: loops through records, calls selected LLM client, fills `output`.                                          | Async aware; supports batch calls.                            |
| **`core/scoring/*`**               | One module per scorer. All expose `def score(record: EvalRecord, cfg: Any) -> Score`.                                  | Register in `scoring.__init__` for dynamic listing.           |
| **`core/evaluation.py`**           | `run_evaluation(records, scorer_cfgs) -> RunResult`. Handles concurrency, retries, logging.                            | Keeps Streamlit thread clear; progress reported via callback. |
| **`core/reporting.py`**            | Aggregate stats → dict, plus `to_csv()` / `to_json()`.                                                                 | Consumed by UI & download page.                               |
| **`services/llm_clients.py`**      | Thin, typed wrappers around vendor SDKs. Standard interface: `generate(prompt, **params)`; `evaluate()` for judge LLM. | Centralizes retry logic, rate limits, exponential back-off.   |
| **`utils/file_cache.py`**          | Optional local caching for expensive LLM calls (dev mode).                                                             | Simple JSON-on-disk keyed by hash of call.                    |
| **`utils/telemetry.py`**           | Early placeholder to push OpenTelemetry spans.                                                                         | Keeps traces optional but path-ready.                         |
| **`logging_config.py`**            | Configures struct-log / standard logging for entire project.                                                           | Import first in `streamlit_app.py`.                           |
| **`tests/`**                       | Pytest suites. Unit tests for every scorer; integration test covers Mode A pipeline with fixtures.                     | CI fails fast on scoring regressions.                         |
| **CI workflow**                    | Lint (`ruff` + `mypy`), run tests.                                                                                     | Container step can later run Streamlit e2e with Playwright.   |

---

## 3 Extensibility & Future Features Hooks

* **New scorer drop-in:** put a `*.py` under `core/scoring/`, define `score()`, add to `__all__` list—UI auto-picks it up because `scoring.list_scorers()` enumerates the modules.
* **Persisted runs & cross-run analytics:** `RunResult` already serializes cleanly; simply store JSON in `/runs/` or a DB.  A future page could load multiple `RunResult` files and feed them to a Plotly comparison view.
* **OpenTelemetry stream:** `utils.telemetry.trace_llm_call()` is invoked in `services.llm_clients.*`.  Switching to a real OTLP exporter later is configuration only.
* **API backend alternative:** If you later need a headless service, everything under `core/` is UI-agnostic.  Wrap it in FastAPI without touching Streamlit pages.

---

## 4 Immediate Next Steps

1. **Scaffold repo** with the tree above (`cookiecutter` or `copier` template).
2. Implement **data models** and **exact-match scorer** first—fastest path to an end-to-end “Hello World” evaluation.
3. Add **fuzzy scorer** (pure Python `python-Levenshtein`).
4. Wire **Streamlit pages** minimally to ingest CSV and call `evaluation.run_evaluation`.
5. Layer in **LLM clients** and **LLM-judge scorer** once the plaintext path is solid.
6. Harden with **unit tests + CI** before tackling Mode B generation.

Feel free to ask for deeper dives on any module, detailed class signatures, or a cookiecutter template.
