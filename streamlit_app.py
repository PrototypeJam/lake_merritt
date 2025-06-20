# git fetch origin
# git checkout restore-lake-merritt
# git pull origin restore-lake-merritt
# rm -rf venv (remove old broken virtual environment)
# uv venv venv (maken good new virtual environment)
# ___ Do Above for Working Snapshot Branch THEN Continue With Below___
# source venv/bin/activate
# uv pip install -e ".[test,dev]"
# uv pip install -r requirements.txt (OLD - test above before deleting this)
# pip install --upgrade pip (maybe not needed?)
# streamlit run streamlit_app.py
# ========================================
# rm -rf venv (if needed)
# uv venv venv
# source venv/bin/activate
# uv pip install -e ".[test,dev]"
# streamlit run streamlit_app.py


"""
AI Evaluation Workbench - Main Application Entry Point
"""
import sys
from pathlib import Path

import streamlit as st

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

# Debug imports
try:
    from core.logging_config import setup_logging

    print("✓ Core imports successful")
except ImportError as e:
    st.error(f"Import error: {e}")
    raise

# Set up logging first
setup_logging()

# Page configuration
st.set_page_config(
    page_title="AI Evaluation Workbench",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Define pages using st.Page
home_page = st.Page(
    "streamlit_app_home.py",
    title="Home",
    icon="🏠",
    default=True,
)

config_page = st.Page(
    "app/pages/1_config.py",
    title="System Configuration",
    icon="⚙️",
)

eval_setup_page = st.Page(
    "app/pages/2_eval_setup.py",
    title="Evaluation Setup",
    icon="📄",
)

results_page = st.Page(
    "app/pages/3_results.py",
    title="View Results",
    icon="📊",
)

downloads_page = st.Page(
    "app/pages/4_downloads.py",
    title="Download Center",
    icon="⬇️",
)

# Create navigation
pg = st.navigation(
    [
        home_page,
        config_page,
        eval_setup_page,
        results_page,
        downloads_page,
    ]
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

# Run the selected page
pg.run()
