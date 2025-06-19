"""
Page 2: Evaluation Setup - Data Upload and Scoring Configuration
"""

import streamlit as st
import pandas as pd
from typing import List, Dict, Any
import asyncio
import nest_asyncio
from io import StringIO
import logging

from core.ingestion import load_evaluation_data, validate_csv_columns
from core.generation import generate_outputs
from core.evaluation import run_evaluation_batch
from core.data_models import EvaluationItem, EvaluationMode
from core.scoring import get_available_scorers
from services.llm_clients import create_llm_client

logger = logging.getLogger(__name__)

st.title("📄 Evaluation Setup")
st.markdown("Upload data, configure evaluation mode, and select scoring methods.")

# Check prerequisites
if not st.session_state.api_keys:
    st.warning("⚠️ Please configure API keys in the System Configuration page first.")
    st.stop()

# Evaluation Mode Selection
st.header("1. Select Evaluation Mode")
mode = st.radio(
    "Choose how you want to evaluate:",
    [EvaluationMode.EVALUATE_EXISTING, EvaluationMode.GENERATE_THEN_EVALUATE],
    format_func=lambda x: (
        "Mode A: Evaluate Existing Outputs"
        if x == EvaluationMode.EVALUATE_EXISTING
        else "Mode B: Generate Outputs, Then Evaluate"
    ),
    horizontal=True,
)

st.session_state.evaluation_mode = mode

# File Upload Section
st.header("2. Upload Evaluation Data")

if mode == EvaluationMode.EVALUATE_EXISTING:
    st.info(
        "Upload a CSV with columns: `input`, `output`, `expected_output` (and optionally `id`)"
    )
else:
    st.info(
        "Upload a CSV with columns: `input`, `expected_output` (and optionally `id`)"
    )

uploaded_file = st.file_uploader(
    "Choose a CSV or JSON file",
    type=["csv", "json"],
    help="Maximum file size: 200MB",
)

if uploaded_file is not None:
    MAX_FILE_SIZE_MB = 100
    if uploaded_file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
        st.error(
            f"❌ File is too large ({uploaded_file.size / 1024**2:.1f}MB). Maximum size is {MAX_FILE_SIZE_MB}MB."
        )
        st.stop()

    try:
        # Check file type and load accordingly
        if uploaded_file.type == "application/json":
            from core.otel.ingester import OTelTraceIngester
            raw_str = uploaded_file.getvalue().decode("utf-8")
            traces = OTelTraceIngester().ingest_str(raw_str)
            st.session_state.eval_data = traces
            st.success(f"✅ Loaded {len(traces)} OTel traces.")
            with st.expander("Trace preview"):
                st.json(traces[0].metadata["otel_trace"])
        else:
            # Load and validate CSV data
            df = pd.read_csv(uploaded_file)

            # Check for empty dataframe
            if df.empty:
                st.error(
                    "❌ The uploaded CSV file is empty. Please provide a file with data."
                )
                st.stop()

            # Validate columns based on mode
            required_cols = ["input", "expected_output"]
            if mode == EvaluationMode.EVALUATE_EXISTING:
                required_cols.append("output")

            is_valid, message = validate_csv_columns(df, required_cols)

            if not is_valid:
                st.error(f"❌ CSV Validation Failed: {message}")
                st.info(
                    f"📋 Required columns for this mode: {', '.join([f'`{col}`' for col in required_cols])}"
                )
                st.info(
                    f"📄 Your file has: {', '.join([f'`{col}`' for col in df.columns.tolist()])}"
                )
                st.stop()

            # Show data preview
            st.success(f"✅ Loaded {len(df)} rows successfully!")

            with st.expander("📊 Data Preview (first 5 rows)"):
                st.dataframe(df.head(), use_container_width=True)

            # Convert to evaluation items
            eval_items = load_evaluation_data(df, mode)
            st.session_state.eval_data = eval_items

    except pd.errors.EmptyDataError:
        st.error("❌ The uploaded file appears to be empty or corrupted.")
        st.info("Please ensure your CSV file contains data and is properly formatted.")
        st.stop()
    except pd.errors.ParserError as e:
        st.error(f"❌ Error parsing CSV file: {str(e)}")
        st.info(
            "💡 Common issues: Inconsistent number of columns per row or unescaped commas within fields."
        )
        st.stop()
    except UnicodeDecodeError:
        st.error(
            "❌ File encoding error. Please ensure your CSV is saved in UTF-8 format."
        )
        st.info("💡 How to fix: In Excel, use 'Save As' and choose 'CSV UTF-8' format.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Unexpected error loading file: {str(e)}")
        logger.exception("Failed to load uploaded file")
        st.stop()

# Mode B: Actor Model Configuration
if mode == EvaluationMode.GENERATE_THEN_EVALUATE and st.session_state.eval_data:
    st.header("3. Configure Actor Model")
    st.markdown("Select the model that will generate outputs for your inputs.")

    col1, col2 = st.columns([1, 2])

    with col1:
        actor_provider = st.selectbox(
            "Actor Model Provider",
            ["openai", "anthropic", "google"],
            key="actor_provider",
        )

        # Model selection based on provider
        model_options = {
            "openai": ["gpt-4", "gpt-4-turbo-preview", "gpt-3.5-turbo"],
            "anthropic": [
                "claude-3-opus-20240229",
                "claude-3-sonnet-20240229",
                "claude-3-haiku-20240307",
            ],
            "google": ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-1.0-pro"],
        }

        actor_model = st.selectbox(
            "Actor Model",
            model_options[actor_provider],
            key="actor_model",
        )

        actor_temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            key="actor_temp",
        )

        actor_max_tokens = st.number_input(
            "Max Tokens",
            min_value=100,
            max_value=4000,
            value=1000,
            step=100,
            key="actor_tokens",
        )

    with col2:
        actor_system_prompt = st.text_area(
            "Actor System Prompt (optional)",
            placeholder="Leave empty to use the input as-is, or provide instructions for how the model should respond",
            height=200,
            key="actor_prompt",
        )

    # Generate outputs button
    if st.button("🚀 Generate Outputs", type="primary", key="generate_btn"):
        with st.spinner("Generating outputs... This may take a few minutes."):
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Create actor configuration
            actor_config = {
                "provider": actor_provider,
                "model": actor_model,
                "temperature": actor_temperature,
                "max_tokens": actor_max_tokens,
                "system_prompt": actor_system_prompt or None,
                "api_key": st.session_state.api_keys.get(actor_provider),
            }

            try:
                # Run generation
                updated_items = asyncio.run(
                    generate_outputs(
                        st.session_state.eval_data,
                        actor_config,
                        progress_callback=lambda i, total: (
                            progress_bar.progress(i / total),
                            status_text.text(f"Processing {i}/{total} items..."),
                        ),
                    )
                )

                st.session_state.eval_data = updated_items
                st.success(
                    f"✅ Successfully generated outputs for {len(updated_items)} items!"
                )

            except Exception as e:
                st.error(f"Error generating outputs: {str(e)}")
                st.stop()

# Scorer Selection Section
if st.session_state.eval_data and (
    mode == EvaluationMode.EVALUATE_EXISTING
    or (
        mode == EvaluationMode.GENERATE_THEN_EVALUATE
        and all(item.output for item in st.session_state.eval_data)
    )
):
    st.header("4. Select Scoring Methods")

    available_scorers = get_available_scorers()

    selected_scorers = st.multiselect(
        "Choose one or more scoring methods:",
        options=list(available_scorers.keys()),
        default=["exact_match"],
        format_func=lambda x: available_scorers[x]["display_name"],
        help="Each scorer will evaluate all items in your dataset",
    )

    st.session_state.selected_scorers = selected_scorers

    # Scorer-specific configuration
    scorer_configs = {}

    for scorer_name in selected_scorers:
        scorer_info = available_scorers[scorer_name]

        with st.expander(f"⚙️ Configure {scorer_info['display_name']}"):
            st.markdown(scorer_info["description"])

            if scorer_name == "fuzzy_match":
                threshold = st.slider(
                    "Similarity Threshold",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.8,
                    step=0.05,
                    help="Minimum similarity score to consider a match",
                    key=f"{scorer_name}_threshold",
                )
                scorer_configs[scorer_name] = {"threshold": threshold}

            elif scorer_name == "llm_judge":
                # Use default judge config or allow override
                use_default = st.checkbox(
                    "Use default judge configuration",
                    value=True,
                    key=f"{scorer_name}_use_default",
                )

                if use_default:
                    scorer_configs[scorer_name] = st.session_state.model_configs[
                        "default_judge_config"
                    ].copy()
                    st.json(scorer_configs[scorer_name])
                else:
                    # Allow custom configuration
                    judge_provider = st.selectbox(
                        "Judge Provider",
                        ["openai", "anthropic", "google"],
                        key=f"{scorer_name}_provider",
                    )

                    model_options = {
                        "openai": ["gpt-4", "gpt-4-turbo-preview", "gpt-3.5-turbo"],
                        "anthropic": [
                            "claude-3-opus-20240229",
                            "claude-3-sonnet-20240229",
                        ],
                        "google": ["gemini-1.5-pro", "gemini-1.5-flash"],
                    }

                    judge_model = st.selectbox(
                        "Judge Model",
                        model_options[judge_provider],
                        key=f"{scorer_name}_model",
                    )

                    judge_temp = st.slider(
                        "Temperature",
                        0.0,
                        1.0,
                        0.3,
                        0.1,
                        key=f"{scorer_name}_temp",
                    )

                    judge_prompt = st.text_area(
                        "Judge Prompt",
                        value=st.session_state.model_configs["default_judge_config"][
                            "system_prompt"
                        ],
                        height=150,
                        key=f"{scorer_name}_prompt",
                    )

                    scorer_configs[scorer_name] = {
                        "provider": judge_provider,
                        "model": judge_model,
                        "temperature": judge_temp,
                        "system_prompt": judge_prompt,
                        "api_key": st.session_state.api_keys.get(judge_provider),
                    }
            else:
                # No configuration needed
                scorer_configs[scorer_name] = {}

    # Run Evaluation Button
    st.header("5. Run Evaluation")

    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        if st.button(
            "🔬 Start Evaluation",
            type="primary",
            use_container_width=True,
            disabled=not selected_scorers,
        ):
            with st.spinner("Running evaluation..."):
                progress_bar = st.progress(0)
                status_text = st.empty()

                try:
                    # DEVELOPER NOTE: This specific pattern for getting the event loop is
                    # the recommended best practice for using asyncio within Streamlit.
                    # It safely gets the existing event loop or creates a new one for the
                    # current thread if one doesn't exist, avoiding the common
                    # `RuntimeError` that `asyncio.run()` can cause. [2, 3]
                    try:
                        loop = asyncio.get_event_loop()
                    except RuntimeError:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)

                    results = loop.run_until_complete(
                        run_evaluation_batch(
                            st.session_state.eval_data,
                            selected_scorers,
                            scorer_configs,
                            st.session_state.api_keys,
                            batch_size=10,
                            progress_callback=lambda i, total: (
                                progress_bar.progress(i / total),
                                status_text.text(f"Evaluating {i}/{total} items..."),
                            ),
                        )
                    )

                    st.session_state.eval_results = results
                    st.success("✅ Evaluation completed successfully!")

                except Exception as e:
                    import logging

                    logger = logging.getLogger(__name__)
                    logger.exception("Evaluation failed")
                    st.error(f"Error during evaluation: {str(e)}")

    with col2:
        st.metric("Total Items", len(st.session_state.eval_data))

    with col3:
        st.metric("Selected Scorers", len(selected_scorers))

# Navigation hints
if not st.session_state.eval_data:
    st.info("👆 Upload a CSV file to begin evaluation setup.")
elif mode == EvaluationMode.GENERATE_THEN_EVALUATE and not all(
    item.output for item in st.session_state.eval_data
):
    st.info("👆 Generate outputs before selecting scorers.")
elif not st.session_state.selected_scorers:
    st.info("👆 Select at least one scoring method to run evaluation.")
