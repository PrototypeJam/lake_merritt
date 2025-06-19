# [Lake Merritt](https://lakemerritt.streamlit.app/): AI Evaluation Workbench

A general-purpose, modular, and extensible platform for evaluating AI models and applications, primarily Large Language Models (LLMs).

## Overview

This platform provides a standardized yet flexible environment for:
- Configuring evaluation parameters and models
- Uploading evaluation datasets or generating model outputs
- Applying multiple scoring methods (exact match, fuzzy match, LLM-as-a-Judge)
- Analyzing results through intuitive visualizations
- Comparing performance across different models and configurations

## Features

### Current (v0)
- **Mode A**: Evaluate existing model outputs against expected outputs
- **Mode B**: Generate outputs from an Actor LLM, then evaluate
- Multiple scoring methods with configurable parameters
- Streamlit-based UI with session state management
- Support for CSV data import/export
- Modular architecture for easy extension

### Planned
- Cross-run analysis and comparison
- Live system monitoring via OpenTelemetry
- Enhanced LLM-as-a-Judge configuration
- Prompt versioning and management

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ai-eval-workbench.git
cd ai-eval-workbench
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
# For development (includes testing and linting tools):
pip install -e ".[test,dev]"

# For standard installation:
pip install .
```

4. Copy `.env.template` to `.env` and add your API keys:
```bash
cp .env.template .env
```

## Usage

Run the Streamlit app:
```bash
streamlit run streamlit_app.py
```

Navigate through the pages:
1. **System Configuration**: Set up API keys and default model parameters
2. **Evaluation Setup**: Upload data, select scoring methods, and run evaluations
3. **View Results**: Analyze evaluation outcomes and detailed scores
4. **Download Center**: Export results in various formats

## Data Format

For Mode A (evaluate existing outputs), your CSV should have:
- `input`: The prompt/input given to the model
- `output`: The model's actual output
- `expected_output`: The ideal/correct output
- `id` (optional): Unique identifier for each row

For Mode B (generate then evaluate), your CSV needs only:
- `input`: The prompt/input for the model
- `expected_output`: The ideal/correct output

## Architecture

The project follows a modular architecture:
- `app/`: Streamlit UI layer
- `core/`: Business logic and evaluation orchestration
- `services/`: External API integrations (LLM providers)
- `utils/`: Helper utilities

### Evaluating OpenTelemetry traces
1. Go to **Evaluation Setup**.
2. Upload `manual_traces.json` (or any JSON in the same schema).
3. Select **Criteria Selection Judge** in the scorer picker.
4. Run. Results will include an expandable timeline for each trace.

## Contributing

This project emphasizes deep modularity. When adding new features:
1. Scorers go in `core/scoring/`
2. LLM providers extend `services/llm_clients.py`
3. UI pages go in `app/pages/`
4. All data structures should be defined as Pydantic models in `core/data_models.py`

## License

MIT License - see LICENSE file for details
