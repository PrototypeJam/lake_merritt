# In file: app/pages/2_eval_setup.py

import asyncio
import logging
import pandas as pd
import streamlit as st
import nest_asyncio
import yaml
import io
from datetime import datetime
import copy

from core.data_models import EvaluationMode
from core.evaluation import run_evaluation_batch
from core.scoring import get_available_scorers
from core.eval_pack.loader import EvalPackLoader
from core.eval_pack.schema import GenerationMode, GenerationConfig, LLMConfig
from core.generation_handler import handle_mode_b_generation, prepare_csv_for_download

logger = logging.getLogger(__name__)

try:
    nest_asyncio.apply()
except RuntimeError:
    pass

st.title("📄 Evaluation Setup")
st.markdown("Upload data, configure evaluation mode, and select scoring methods.")

if not st.session_state.get("api_keys"):
    st.warning("⚠️ Please configure API keys in the System Configuration page first.")
    st.stop()

eval_method = st.radio(
    "Choose evaluation method:",
    ["Configure Manually", "Upload Eval Pack"],
    horizontal=True,
    index=0
)
st.markdown("---")

# ==============================================================================
# SECTION 1: MANUAL WORKFLOW
# ==============================================================================
if eval_method == "Configure Manually":
    st.header("1. Select Evaluation Mode")
    mode = st.radio(
        "Choose what you want to do:",
        [EvaluationMode.EVALUATE_EXISTING, EvaluationMode.GENERATE_THEN_EVALUATE],
        format_func=lambda x: "Mode A: Evaluate Existing Outputs" if x == EvaluationMode.EVALUATE_EXISTING else "Mode B: Generate New Data",
        horizontal=True,
    )
    st.session_state.evaluation_mode = mode

    st.header("2. Upload Data")
    if mode == EvaluationMode.EVALUATE_EXISTING:
        st.info("Upload a CSV with `input`, `output`, and `expected_output` columns.")
    else: # Mode B
        st.info("Upload a CSV with an `input` column. For 'Generate Outputs' mode, it must also include an `expected_output` column.")

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"], key="manual_upload")

    if uploaded_file:
        try:
            for key in ['items_to_evaluate', 'generated_items', 'data_is_ready_for_scoring', 'raw_data_for_manual_eval']:
                if key in st.session_state:
                    del st.session_state[key]
            
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded `{uploaded_file.name}`. Showing preview:")
            with st.expander("📊 Data Preview"):
                st.dataframe(df.head(), use_container_width=True)
            uploaded_file.seek(0)
            st.session_state.raw_data_for_manual_eval = uploaded_file
        except Exception as e:
            st.error(f"Could not read or preview CSV. Error: {e}")
            if 'raw_data_for_manual_eval' in st.session_state:
                del st.session_state.raw_data_for_manual_eval
            st.stop()

    # --- MODE B: GENERATION WORKFLOW ---
    if mode == EvaluationMode.GENERATE_THEN_EVALUATE and 'raw_data_for_manual_eval' in st.session_state:
        st.header("3. Configure Mode B Generation")
        
        sub_mode_tuple = st.radio(
            "What would you like to generate?",
            options=[
                ("Generate Model Outputs (to evaluate them)", GenerationMode.GENERATE_OUTPUTS), 
                ("Generate Expected Outputs (to create a dataset)", GenerationMode.GENERATE_EXPECTED_OUTPUTS)
            ],
            format_func=lambda x: x[0],
            key="generation_mode_selection"
        )
        sub_mode = sub_mode_tuple[1]

        st.subheader("📝 Provide High-Level Context")
        user_context = st.text_area("Enter context to guide the AI's generation task (e.g., persona, style, constraints):", height=150, key="generation_context_text")

        # FIX: The model configuration and generate button are now always visible in Mode B,
        # but the button is disabled until context is provided. This fixes the "missing button" bug.
        st.subheader("🤖 Configure Generation Model")
        g_col1, g_col2 = st.columns(2)
        with g_col1:
            gen_provider = st.selectbox("Provider", ["openai", "anthropic", "google"], key="gen_provider")
            model_options = { "openai": ["gpt-4o", "gpt-4-turbo"], "anthropic": ["claude-3-sonnet-20240229"], "google": ["gemini-1.5-pro"] }
            gen_model = st.selectbox("Model", model_options.get(gen_provider, []), key="gen_model")
        with g_col2:
            gen_temp = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1, key="gen_temp")
            gen_tokens = st.number_input("Max Tokens", 100, 4000, 512, 50, key="gen_tokens")

        can_generate = st.session_state.api_keys.get(gen_provider) is not None
        if not can_generate:
             st.error(f"API key for {gen_provider} is not configured in System Configuration.")

        if st.button("🚀 Generate Data", type="primary", disabled=(not can_generate or not user_context.strip())):
            with st.spinner("Generating data... This may take a few moments."):
                try:
                    gen_config = GenerationConfig(mode=sub_mode, data_generator_llm=LLMConfig(provider=gen_provider, model=gen_model, temperature=gen_temp, max_tokens=gen_tokens))
                    
                    items, metadata = asyncio.run(handle_mode_b_generation(
                        st.session_state.raw_data_for_manual_eval, gen_config, user_context, st.session_state.api_keys
                    ))
                    
                    st.session_state.generated_items = items
                    st.session_state.generation_sub_mode = sub_mode
                    st.success(f"Successfully generated data for {metadata['successful_generations']}/{metadata['total_items']} items.")
                    if metadata['failed_generations'] > 0:
                        st.warning(f"{metadata['failed_generations']} items failed to generate. Check the downloaded CSV for '[ERROR]' messages.")
                except Exception as e:
                    st.error(f"Generation failed: {e}")

        if 'generated_items' in st.session_state:
            st.header("4. Review and Use Generated Data")
            df = pd.read_csv(io.StringIO(prepare_csv_for_download(st.session_state.generated_items)))
            st.dataframe(df.head(), use_container_width=True)
            st.download_button("📥 Download Full CSV", prepare_csv_for_download(st.session_state.generated_items), f"generated_data_{datetime.now().strftime('%Y%m%d%H%M')}.csv", "text/csv")

            if st.session_state.generation_sub_mode == GenerationMode.GENERATE_OUTPUTS:
                if st.button("📊 Proceed to Evaluation", type="secondary"):
                    st.session_state.items_to_evaluate = st.session_state.generated_items
                    st.session_state.data_is_ready_for_scoring = True
                    st.rerun() 
            else:
                st.info("Your dataset with generated expected outputs is ready for download. To evaluate a model against it, start a new 'Mode A' evaluation with the downloaded file.")

    # --- SCORING AND EVALUATION UI ---
    is_ready_for_scoring = (mode == EvaluationMode.EVALUATE_EXISTING and 'raw_data_for_manual_eval' in st.session_state) or \
                           st.session_state.get('data_is_ready_for_scoring', False)

    if is_ready_for_scoring:
        if 'items_to_evaluate' not in st.session_state:
            try:
                from core.ingestion.csv_ingester import CSVIngester
                st.session_state.items_to_evaluate = CSVIngester().ingest(st.session_state.raw_data_for_manual_eval, {})
            except Exception as e:
                st.error(f"Error preparing data for evaluation: {e}")
                st.stop()
        
        st.header("5. Configure Scoring Methods")
        available_scorers = get_available_scorers()
        selected_scorers = st.multiselect("Select scorers:", list(available_scorers.keys()), default=["exact_match", "llm_judge"], format_func=lambda x: available_scorers[x]['display_name'])
        
        scorer_configs = {}
        for scorer in selected_scorers:
            with st.expander(f"⚙️ Configure {available_scorers[scorer]['display_name']}"):
                if scorer == "fuzzy_match":
                    scorer_configs[scorer] = {"threshold": st.slider("Threshold", 0.0, 1.0, 0.8, 0.05, key=f"{scorer}_thresh")}
                elif scorer == "llm_judge":
                    cfg = copy.deepcopy(st.session_state.model_configs["default_judge_config"])
                    
                    cfg['provider'] = st.selectbox("Provider", ["openai", "anthropic", "google"], index=["openai", "anthropic", "google"].index(cfg['provider']), key=f"{scorer}_provider")
                    model_options = { "openai": ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"], "anthropic": ["claude-3-5-sonnet-20240620", "claude-3-opus-20240229"], "google": ["gemini-1.5-pro", "gemini-1.5-flash"] }
                    cfg['model'] = st.selectbox("Model", model_options.get(cfg['provider'], [cfg['model']]), key=f"{scorer}_model")
                    cfg['temperature'] = st.slider("Temperature", 0.0, 1.0, cfg['temperature'], 0.1, key=f"{scorer}_temp")
                    cfg['user_prompt_template'] = st.text_area("User Prompt Template", cfg.get('user_prompt_template', ''), height=250, key=f"{scorer}_template")
                    
                    scorer_configs[scorer] = cfg
                else:
                    scorer_configs[scorer] = {}
        
        st.header("6. Run Evaluation")
        if st.button("🔬 Start Evaluation", type="primary", use_container_width=True):
            with st.spinner("Running evaluation..."):
                try:
                    from core.eval_pack.compatibility import create_legacy_pack
                    pack = create_legacy_pack(selected_scorers, scorer_configs, st.session_state.api_keys)
                    
                    results = asyncio.run(run_evaluation_batch(
                        items=st.session_state.items_to_evaluate,
                        pack=pack,
                        api_keys=st.session_state.api_keys
                    ))
                    
                    st.session_state.eval_results = results
                    st.success("Evaluation complete! Redirecting to results page...")
                    st.switch_page("app/pages/3_results.py")
                except Exception as e:
                    logger.exception("Manual evaluation failed")
                    st.error(f"Error during evaluation: {e}")

# ==============================================================================
# SECTION 2: EVAL PACK WORKFLOW
# ==============================================================================
else: 
    # This section remains unchanged as it was not part of the bug report.
    st.header("1. Upload Eval Pack")
    uploaded_pack_file = st.file_uploader("Upload an Eval Pack (.yaml or .yml)", type=['yaml', 'yml'], key="pack_upload")
    
    if uploaded_pack_file:
        try:
            pack_loader = EvalPackLoader()
            pack_content = uploaded_pack_file.getvalue().decode('utf-8')
            pack_dict = yaml.safe_load(pack_content)
            pack, validation_errors = pack_loader.load(source=pack_dict)
            
            if validation_errors:
                st.error("❌ Eval Pack validation failed:")
                for error in validation_errors: st.error(f"• {error}")
                st.stop()

            st.session_state.pack = pack
            st.success(f"✅ Loaded and validated Eval Pack: **{pack.name}**")
            with st.expander("Pack Details", expanded=False):
                st.json(pack.model_dump_json(indent=2))
                        
        except Exception as e:
            st.error(f"❌ Error loading Eval Pack: {str(e)}")
            logger.exception("Failed to load eval pack")
            st.stop()
            
    if 'pack' in st.session_state:
        pack = st.session_state.pack
        st.header("2. Upload Data")
        data_file = st.file_uploader(f"Upload data compatible with '{pack.ingestion.type}' ingester", key="pack_data_upload")
        
        user_context_pack = None
        if pack.generation:
            st.subheader("📝 Provide Context for Generation")
            st.info("This pack uses Mode B. Provide high-level context to guide the data generation.")
            user_context_pack = st.text_area("Enter context to guide generation:", height=200, key="pack_context")

        if data_file:
            st.header("3. Run")
            if st.button("🔬 Start Pack Run", type="primary", use_container_width=True):
                if pack.generation and not user_context_pack:
                    st.error("Please provide context for the generation step.")
                    st.stop()

                with st.spinner(f"Running pack '{pack.name}'..."):
                    try:
                        loop = asyncio.get_event_loop()
                        results = loop.run_until_complete(run_evaluation_batch(
                            raw_data=data_file,
                            pack=pack,
                            api_keys=st.session_state.api_keys,
                            user_context=user_context_pack
                        ))
                        st.session_state.eval_results = results
                        st.success("✅ Pack run completed successfully!")
                        st.switch_page("app/pages/3_results.py")
                    except Exception as e:
                        logger.exception("Pack evaluation failed")
                        st.error(f"Error during pack run: {str(e)}")