# AGENTS.md

## Lake Merritt AI Evaluation Workbench

You are **OpenAI Codex** operating in code-mode.

**Project goal**   
Implement and evolve the Lake Merritt **“Eval Packs”** architecture (see `docs/dev_plan.md`) **and complete the numbered fixes in `docs/FixList.md`.**

---

### Scope and Hierarchy
* **AGENTS.md** applies to the entire repository.  
* **Task-specific instructions in `docs/FixList.md` take precedence** for fix-related work.

---

## Workflow

### Task Execution Protocol
1. **Read the full `FixPlan.md` / `FixList.md`** to understand context.  
2. **Work on ONE numbered task at a time** as directed by the user.  
3. Make only the changes required for that task.  
4. Run **all programmatic checks** (see below).  
5. Open a focused PR titled `fix/<task-id>: <slug>` that references the task number.  
6. Ensure no unrelated changes are included.

### Pull-Request Guidelines
* **Title format**: `[fix/<task-id>] Brief description`  
  Example: `fix/02: Hard-code CSV override for manual mode`.
* **Description must include**  
  – Reference to the task in `docs/FixList.md`  
  – Summary of changes  
  – Test results / coverage output  
  – Deviations from plan (and why)
* **Before opening PR**  
  – Run all checks below  
  – Squash or re-base to keep history clean

---

## Environment Setup
This project runs on **Python 3.13** (3.9+ minimum).  
Dependencies are declared in `pyproject.toml`; use **`uv`** for fast, reproducible installs.

```bash
# First-time setup
uv pip install -e ".[test,dev]"
````

---

## Testing Guidelines

**IMPORTANT** – skip API-dependent tests.

```bash
# Run the safe suite (unit + integration, no API)
pytest -v -m "not requires_api"

# Run the new integration test explicitly
pytest tests/integration/test_eval_pack_json_ingestion.py -v
```

* Unit tests live in `tests/unit/`
* Integration tests live in `tests/integration/`
* Tests requiring external keys are marked `@pytest.mark.requires_api` and are skipped in CI.

### Programmatic Checks (must pass before PR)

```bash
# 1. Safe tests
pytest -v -m "not requires_api"

# 2. mypy type check
mypy core --ignore-missing-imports

# 3. Black formatting
black --check core tests

# 4. Import sanity
python -c "import core; print('Core imports successfully')"
```

---

## Code Style & Generation

* Format with **Black**; run `black core tests` to auto-format.
* **Type-hints required** for all new functions.
* Docstrings follow the **Google Python Style Guide**.
* Avoid placeholders—when you replace a function, include the entire body.
* Preserve existing behaviour unless the task explicitly changes it.
* Add clear inline comments for non-obvious logic, following patterns already in the codebase.

---

## Quick Reference

```bash
# Install deps (fallback to pip if uv unavailable)
uv pip install -e ".[test,dev]" || pip install -e ".[test,dev]"

# Run safe tests
pytest -v -m "not requires_api"

# Format code
black core tests
```

---

### Reminders

* **Never commit API keys or `.env` files.**
* The Streamlit UI requires manual testing; CI focuses on `core/` logic.
* Keep PRs granular—one fix = one PR.