"""
AI Evaluation Workbench - Main Application Entry Point
"""
import streamlit as st
from pathlib import Path
import sys

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from core.logging_config import setup_logging

# Set up logging first
setup_logging()

# Page configuration
st.set_page_config(
    page_title="AI Evaluation Workbench",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state
if "initialized" not in st.session_state:
    st.session_state.initialized = True
    st.session_state.api_keys = {}
    st.session_state.model_configs = {}
    st.session_state.eval_data = None
    st.session_state.eval_results = None
    st.session_state.selected_scorers = []
    st.session_state.run_metadata = {}

# Main page content
st.title("🔬 AI Evaluation Workbench")
st.markdown("---")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    Welcome to the AI Evaluation Workbench, a modular platform for evaluating 
    Large Language Models (LLMs) and AI applications.
    
    ### Getting Started
    
    1. **Configure System** - Set up your API keys and default model parameters
    2. **Setup Evaluation** - Upload data and select scoring methods
    3. **View Results** - Analyze evaluation outcomes
    4. **Download Results** - Export data for further analysis
    
    ### Evaluation Modes
    
    - **Mode A**: Evaluate pre-existing model outputs against expected outputs
    - **Mode B**: Generate outputs from a model, then evaluate them
    
    Use the sidebar to navigate between pages.
    """)

with col2:
    st.info("""
    **Quick Tips:**
    - Start with the System Configuration page
    - Prepare your CSV data in the required format
    - Multiple scorers can be applied simultaneously
    - Results are preserved during your session
    """)

# Status dashboard
st.markdown("### Current Status")
status_cols = st.columns(4)

with status_cols[0]:
    api_configured = len(st.session_state.api_keys) > 0
    st.metric(
        "API Configuration",
        "\u2705 Configured" if api_configured else "\u274C Not Set",
        delta=None,
    )

with status_cols[1]:
    data_loaded = st.session_state.eval_data is not None
    st.metric(
        "Data Loaded",
        "\u2705 Ready" if data_loaded else "\u274C No Data",
        delta=None,
    )

with status_cols[2]:
    scorers_selected = len(st.session_state.selected_scorers) > 0
    st.metric(
        "Scorers Selected",
        f"\u2705 {len(st.session_state.selected_scorers)}" if scorers_selected else "\u274C None",
        delta=None,
    )

with status_cols[3]:
    results_available = st.session_state.eval_results is not None
    st.metric(
        "Results",
        "\u2705 Available" if results_available else "\u274C Not Run",
        delta=None,
    )

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        AI Evaluation Workbench v0.1.0 | 
        <a href='https://github.com/yourusername/ai-eval-workbench'>GitHub</a>
    </div>
    """,
    unsafe_allow_html=True,
)
