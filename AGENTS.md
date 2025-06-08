# AGENTS.md

## Lake Merritt AI Evaluation Workbench

### Environment Setup

This project requires Python 3.9+ and uses pyproject.toml for dependency management.

### Testing Guidelines

1. **Unit tests only** - Do not run integration tests that require API keys
   ```bash
   pytest tests/unit -v
