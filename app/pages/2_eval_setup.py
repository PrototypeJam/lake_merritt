# In file: app/pages/2_eval_setup.py
"""
Page 2: Evaluation Setup - Data Upload and Scoring Configuration
"""
import asyncio
import logging
import pandas as pd
import streamlit as st
import nest_asyncio
import yaml
import io  # For completeness, in case you use Option 2B elsewhere

from core.data_models import EvaluationMode
from core.evaluation import run_evaluation_batch
from core.generation import generate_outputs
from core.scoring import get_available_scorers
from core.eval_pack.loader import EvalPackLoader

logger = logging.getLogger(__name__)

# This is a recommended practice for using asyncio within Streamlit
try:
    nest_asyncio.apply()
except RuntimeError:
    pass # It's already applied

st.title("📄 Evaluation Setup")
st.markdown("Upload data, configure evaluation mode, and select scoring methods.")

# Check prerequisites
if not st.session_state.api_keys:
    st.warning("⚠️ Please configure API keys in the System Configuration page first.")
    st.stop()

# --- UI to switch between manual and pack-based evaluation ---
eval_method = st.radio(
    "Choose evaluation method:",
    ["Configure Manually (Legacy)", "Upload Eval Pack (New)"],
    horizontal=True,
    index=0 # Default to the manual configuration
)
st.markdown("---")

# ==============================================================================
# SECTION 1: MANUAL (LEGACY) WORKFLOW
# ==============================================================================
if eval_method == "Configure Manually (Legacy)":
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

    st.header("2. Upload Evaluation Data")
    if mode == EvaluationMode.EVALUATE_EXISTING:
        st.info("Upload a CSV with columns: `input`, `output`, `expected_output` (and optionally `id`)")
    else:
        st.info("Upload a CSV with columns: `input`, `expected_output` (and optionally `id`)")

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        key="manual_upload"
    )

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded `{uploaded_file.name}`. Showing preview:")
            with st.expander("📊 Data Preview (first 5 rows)"):
                st.dataframe(df.head(), use_container_width=True)
            uploaded_file.seek(0)
            st.session_state.raw_data_for_manual_eval = uploaded_file
        except Exception as e:
            st.error(f"Could not read or preview CSV. Error: {e}")
            st.stop()
    
    if mode == EvaluationMode.GENERATE_THEN_EVALUATE and 'raw_data_for_manual_eval' in st.session_state:
        st.header("3. Configure Actor Model")
        col1, col2 = st.columns([1, 2])
        with col1:
            actor_provider = st.selectbox("Actor Model Provider", ["openai", "anthropic", "google"], key="actor_provider")
            model_options = {"openai": ["gpt-4", "gpt-3.5-turbo"], "anthropic": ["claude-3-opus-20240229"], "google": ["gemini-1.5-pro"]}
            actor_model = st.selectbox("Actor Model", model_options[actor_provider], key="actor_model")
            actor_temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1, key="actor_temp")
        with col2:
            actor_system_prompt = st.text_area("Actor System Prompt (optional)", height=200, key="actor_prompt")

        if st.button("🚀 Generate Outputs", type="secondary"):
             with st.spinner("Generating outputs..."):
                from core.ingestion.csv_ingester import CSVIngester
                ingester = CSVIngester()
                initial_items = ingester.ingest(st.session_state.raw_data_for_manual_eval, {"mode": mode})

                actor_config = {
                    "provider": actor_provider, "model": actor_model, "temperature": actor_temperature, 
                    "system_prompt": actor_system_prompt, "api_key": st.session_state.api_keys.get(actor_provider)
                }
                
                loop = asyncio.get_event_loop()
                updated_items = loop.run_until_complete(generate_outputs(initial_items, actor_config))
                st.session_state.generated_items_for_manual_eval = updated_items
                st.success(f"✅ Generated outputs for {len(updated_items)} items. You can now select scorers.")

    st.header("4. Select Scoring Methods")
    data_is_ready = ('raw_data_for_manual_eval' in st.session_state and mode == EvaluationMode.EVALUATE_EXISTING) or \
                    ('generated_items_for_manual_eval' in st.session_state and mode == EvaluationMode.GENERATE_THEN_EVALUATE)

    if data_is_ready:
        available_scorers = get_available_scorers()
        selected_scorers = st.multiselect(
            "Choose one or more scoring methods:",
            options=list(available_scorers.keys()),
            default=["exact_match"],
            format_func=lambda x: available_scorers[x]["display_name"],
        )
        scorer_configs = {}
        for scorer_name in selected_scorers:
            with st.expander(f"⚙️ Configure {available_scorers[scorer_name]['display_name']}"):
                if scorer_name == "fuzzy_match":
                    threshold = st.slider("Similarity Threshold", 0.0, 1.0, 0.8, 0.05, key=f"{scorer_name}_threshold")
                    scorer_configs[scorer_name] = {"threshold": threshold}
                elif scorer_name == "llm_judge":
                    # --- FULLY EDITABLE LLM JUDGE CONFIG ---
                    judge_cfg = st.session_state.model_configs["default_judge_config"].copy()

                    judge_cfg["provider"] = st.selectbox(
                        "LLM Judge Provider",
                        ["openai", "anthropic", "google"],
                        index=["openai", "anthropic", "google"].index(judge_cfg.get("provider", "openai")),
                        key="manual_judge_provider"
                    )

                    model_options = {
                        "openai": [
                            "gpt-4.1",
                            "gpt-4.1-mini",
                            "gpt-4.1-nano",
                            "gpt-4o",
                            "gpt-4-turbo",
                            "gpt-3.5-turbo",
                        ],
                        "anthropic": [
                            "claude-opus-4-20250514",
                            "claude-sonnet-4-20250514",
                            "claude-3-5-sonnet-20240620",
                            "claude-3-5-haiku-20241022",
                            "claude-3-opus-20240229",
                            "claude-3-sonnet-20240229",
                            "claude-3-haiku-20240307",
                        ],
                        "google": [
                            "gemini-2.5-pro",
                            "gemini-2.5-flash",
                            "gemini-2.5-flash-lite-preview-06-17",
                            "gemini-2.5-flash-preview-native-audio-dialog",
                            "gemini-2.5-flash-exp-native-audio-thinking-dialog",
                            "gemini-2.5-flash-preview-tts",
                            "gemini-2.5-pro-preview-tts",
                            "gemini-2.0-flash",
                            "gemini-2.0-flash-preview-image-generation",
                            "gemini-2.0-flash-lite",
                            "gemini-1.5-pro",
                            "gemini-1.5-flash",
                            "gemini-1.5-flash-8b",
                        ],
                    }
                    current_models = model_options[judge_cfg["provider"]]
                    if judge_cfg["model"] not in current_models:
                        judge_cfg["model"] = current_models[0]
                    judge_cfg["model"] = st.selectbox(
                        "LLM Judge Model",
                        current_models,
                        index=current_models.index(judge_cfg["model"]),
                        key="manual_judge_model"
                    )

                    judge_cfg["temperature"] = st.slider(
                        "LLM Judge Temperature",
                        min_value=0.0,
                        max_value=1.0,
                        value=judge_cfg.get("temperature", 0.3),
                        step=0.1,
                        key="manual_judge_temp"
                    )

                    judge_cfg["max_tokens"] = st.number_input(
                        "LLM Judge Max Tokens",
                        min_value=100,
                        max_value=4000,
                        value=judge_cfg.get("max_tokens", 1000),
                        step=100,
                        key="manual_judge_max_tokens"
                    )

                    judge_cfg["threshold"] = st.slider(
                        "LLM Judge Pass Threshold",
                        min_value=0.0,
                        max_value=1.0,
                        value=judge_cfg.get("threshold", 0.7),
                        step=0.05,
                        key="manual_judge_threshold"
                    )

                    judge_cfg["system_prompt"] = st.text_area(
                        "LLM Judge System Prompt",
                        value=judge_cfg.get("system_prompt", ""),
                        height=200,
                        key="manual_judge_system_prompt"
                    )

                    judge_cfg["user_prompt_template"] = st.text_area(
                        "LLM Judge User Prompt Template",
                        value=judge_cfg.get("user_prompt_template", """
Compare the actual output to the expected output for the given input.

Input: {input}
Expected Output: {expected_output}
Actual Output: {output}

Respond in JSON format with:
- "score": 0.0 to 1.0
- "reasoning": explanation of your evaluation
""".strip()),
                        height=200,
                        key="manual_judge_user_prompt"
                    )

                    # --- CRITICAL: Assign API key for selected provider ---
                    judge_cfg["api_key"] = st.session_state.api_keys.get(judge_cfg["provider"])

                    scorer_configs[scorer_name] = judge_cfg
                else:
                    scorer_configs[scorer_name] = {}

        st.header("5. Run Evaluation")
        if st.button("🔬 Start Manual Evaluation", type="primary"):
            with st.spinner("Running manual evaluation..."):
                try:
                    loop = asyncio.get_event_loop()
                    if mode == EvaluationMode.GENERATE_THEN_EVALUATE:
                        # Pass the already-generated items directly
                        results = loop.run_until_complete(
                            run_evaluation_batch(
                                items=st.session_state.generated_items_for_manual_eval,
                                selected_scorers=selected_scorers,
                                scorer_configs=scorer_configs,
                                api_keys=st.session_state.api_keys,
                            )
                        )
                    else:
                        # Pass the raw file for ingestion
                        results = loop.run_until_complete(
                            run_evaluation_batch(
                                raw_data=st.session_state.raw_data_for_manual_eval,
                                selected_scorers=selected_scorers,
                                scorer_configs=scorer_configs,
                                api_keys=st.session_state.api_keys,
                            )
                        )
                    st.session_state.eval_results = results
                    st.success("✅ Manual evaluation completed successfully!")
                except Exception as e:
                    logger.exception("Manual evaluation failed")
                    st.error(f"Error during manual evaluation: {str(e)}")

# ==============================================================================
# SECTION 2: NEW EVAL PACK WORKFLOW
# ==============================================================================
else: # eval_method == "Upload Eval Pack (New)"
    st.header("1. Upload Eval Pack")
    uploaded_pack_file = st.file_uploader("Upload an Eval Pack (.yaml or .yml)", type=['yaml', 'yml'])
    
    pack = None
    if uploaded_pack_file:
        try:
            pack_loader = EvalPackLoader()
            pack_content = uploaded_pack_file.getvalue().decode('utf-8')
            
            # First validate YAML syntax
            try:
                pack_dict = yaml.safe_load(pack_content)
            except yaml.YAMLError as e:
                st.error("❌ Invalid YAML syntax in Eval Pack:")
                st.code(str(e), language='text')
                st.info("💡 Check for incorrect indentation, missing colons, or invalid characters")
                st.stop()
            
            # Then validate pack structure and components
            pack, validation_errors = pack_loader.load(source=pack_dict)
            
            if validation_errors:
                st.error("❌ Eval Pack validation failed:")
                for error in validation_errors:
                    st.error(f"• {error}")
                
                # Provide helpful hints based on common errors
                error_text = " ".join(validation_errors).lower()
                if "unknown ingester" in error_text:
                    st.info("💡 Available ingesters: csv, json, generic_otel, otel, openinference")
                elif "unknown scorer" in error_text:
                    st.info("💡 Available scorers: exact_match, fuzzy_match, llm_judge, tool_usage, criteria_selection_judge")
                
                st.stop()
            
            # Only set pack in session state if fully valid
            st.session_state.pack = pack
            st.success(f"✅ Loaded and validated Eval Pack: **{pack.name}** (v{pack.version})")
            
            # Show pack details for confirmation
            with st.expander("Pack Details", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Ingestion Type", pack.ingestion.type)
                    st.metric("Pipeline Stages", len(pack.pipeline))
                with col2:
                    st.metric("Schema Version", pack.schema_version)
                    if pack.description:
                        st.info(pack.description)
                        
        except Exception as e:
            st.error(f"❌ Unexpected error loading Eval Pack: {str(e)}")
            st.info("💡 Please check the pack file format and try again")
            logger.exception("Failed to load eval pack")
            st.stop()
            
    # This check now uses the pack object that was validated and stored in session_state
    if st.session_state.get("pack"):
        pack = st.session_state.get("pack")
        st.header("2. Upload Data for the Pack")
        data_file = st.file_uploader(f"Upload data compatible with '{pack.ingestion.type}' ingester")
        
        if data_file:
            st.header("3. Run Evaluation")
            if st.button("🔬 Start Pack Evaluation", type="primary"):
                with st.spinner(f"Running evaluation with pack '{pack.name}'..."):
                    try:
                        loop = asyncio.get_event_loop()
                        results = loop.run_until_complete(
                            run_evaluation_batch(
                                raw_data=data_file,
                                pack=pack,
                                api_keys=st.session_state.api_keys # <<< THIS IS THE FIX
                            )
                        )
                        st.session_state.eval_results = results
                        st.success("✅ Pack evaluation completed successfully!")
                    except Exception as e:
                        logger.exception("Pack evaluation failed")
                        st.error(f"Error during pack evaluation: {str(e)}")
