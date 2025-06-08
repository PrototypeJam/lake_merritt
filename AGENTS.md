# AGENTS.md

## Lake Merritt AI Evaluation Workbench

### Environment Setup
This project requires Python 3.9+ and uses requirements.txt for dependency management.
We use `uv` for fast, reliable dependency installation.

### Testing Guidelines
1. **Unit tests only** - Do not run integration tests that require API keys
   ```bash
   pytest tests/unit -v
   ```

2. **Skip LLM tests** - These require API keys not available in CI
   ```bash
   pytest -v -m "not requires_api"
   ```

### Code Style
- Use Black for formatting
- Type hints are required for all new functions
- Docstrings follow Google style

### Common Tasks
- **Install dependencies**: `uv pip install -r requirements.txt`
- **Run a specific scorer test**: `pytest tests/unit/test_exact_match.py -v`
- **Check types**: `mypy core --ignore-missing-imports`
- **Format code**: `black core tests`

### Important Notes
- Do NOT commit API keys or .env files
- The Streamlit app requires manual testing (not suitable for automated CI)
- Focus test efforts on the `core/` module business logic
- If uv is not available, fallback to regular pip
