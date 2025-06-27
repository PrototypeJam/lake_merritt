"""
Page 2: Evaluation Setup - Data Upload and Scoring Configuration
"""

# In file: app/pages/2_eval_setup.py
"""
Page 2: Evaluation Setup - Data Upload and Scoring Configuration
"""
import asyncio
import logging
import pandas as pd
import streamlit as st
import nest_asyncio

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

    # This block now only reads for preview and stores the raw file. No ingestion.
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
    
    # Mode B: Actor Model Configuration - This restores the generation step
    if mode == EvaluationMode.GENERATE_THEN_EVALUATE and 'raw_data_for_manual_eval' in st.session_state:
        st.header("3. Configure Actor Model")
        # Actor model configuration UI (unchanged from original file)
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
                
                # Run generation
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
        # Scorer config UI (this can be copied from the original file)
        scorer_configs = {}
        for scorer_name in selected_scorers:
            with st.expander(f"⚙️ Configure {available_scorers[scorer_name]['display_name']}"):
                if scorer_name == "fuzzy_match":
                    threshold = st.slider("Similarity Threshold", 0.0, 1.0, 0.8, 0.05, key=f"{scorer_name}_threshold")
                    scorer_configs[scorer_name] = {"threshold": threshold}
                elif scorer_name == "llm_judge":
                    scorer_configs[scorer_name] = st.session_state.model_configs["default_judge_config"].copy()
                    st.json(scorer_configs[scorer_name], expanded=False)
                else:
                    scorer_configs[scorer_name] = {}

        st.header("5. Run Evaluation")
        if st.button("🔬 Start Manual Evaluation", type="primary"):
            with st.spinner("Running manual evaluation..."):
                try:
                    # Determine the data source
                    if mode == EvaluationMode.GENERATE_THEN_EVALUATE:
                        # Ingest from the already-generated items in memory
                        data_source = st.session_state.generated_items_for_manual_eval
                    else:
                        # Pass the raw file for ingestion by the core engine
                        data_source = st.session_state.raw_data_for_manual_eval

                    loop = asyncio.get_event_loop()
                    results = loop.run_until_complete(
                        run_evaluation_batch(
                            raw_data=data_source,
                            selected_scorers=selected_scorers,
                            scorer_configs=scorer_configs,
                            api_keys=st.session_state.api_keys,
                            # pack=None is implicit, which triggers the compatibility layer
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
            # Handle potential different ways of reading file content
            if hasattr(uploaded_pack_file, 'getvalue'):
                pack_content = uploaded_pack_file.getvalue().decode('utf-8')
            else:
                pack_content = uploaded_pack_file.read().decode('utf-8')
            
            # Use yaml.safe_load for parsing
            import yaml
            pack_dict = yaml.safe_load(pack_content)
            pack = pack_loader.load_from_dict(pack_dict)

            st.session_state.pack = pack
            st.success(f"Loaded Eval Pack: **{pack.name}** (v{pack.version})")
            with st.expander("Pack Details"):
                st.json(pack.model_dump_json(indent=2))
        except Exception as e:
            st.error(f"Error loading or parsing Eval Pack: {e}")
            st.stop()
            
    if pack:
        st.header("2. Upload Data for the Pack")
        # The data uploader now accepts any file type, as the ingester will handle it
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
                                api_keys=st.session_state.api_keys
                            )
                        )
                        st.session_state.eval_results = results
                        st.success("✅ Pack evaluation completed successfully!")
                    except Exception as e:
                        logger.exception("Pack evaluation failed")
                        st.error(f"Error during pack evaluation: {str(e)}")
