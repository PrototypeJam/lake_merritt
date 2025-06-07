"""
Page 1: System & Model Configuration
"""
import streamlit as st
from typing import Dict, Any
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

st.set_page_config(page_title="System Configuration", page_icon="\u2699\uFE0F", layout="wide")

st.title("\u2699\uFE0F System & Model Configuration")
st.markdown("Configure API keys and default model parameters for evaluations.")

# API Keys Section
st.header("1. API Keys")
st.info("Your API keys are stored only for this session and are not saved to disk.")

col1, col2 = st.columns(2)

with col1:
    openai_key = st.text_input(
        "OpenAI API Key",
        value=st.session_state.api_keys.get("openai", ""),
        type="password",
        help="Required for GPT models",
    )
    if openai_key:
        st.session_state.api_keys["openai"] = openai_key

    anthropic_key = st.text_input(
        "Anthropic API Key",
        value=st.session_state.api_keys.get("anthropic", ""),
        type="password",
        help="Required for Claude models",
    )
    if anthropic_key:
        st.session_state.api_keys["anthropic"] = anthropic_key

with col2:
    google_key = st.text_input(
        "Google AI API Key",
        value=st.session_state.api_keys.get("google", ""),
        type="password",
        help="Required for Gemini models",
    )
    if google_key:
        st.session_state.api_keys["google"] = google_key

# Default Model Configuration
st.header("2. Default Model Configuration")
st.markdown("Set default parameters for LLM-as-a-Judge and other model operations.")

# Initialize default configs if not present
if "default_judge_config" not in st.session_state.model_configs:
    st.session_state.model_configs["default_judge_config"] = {
        "provider": "openai",
        "model": "gpt-4",
        "temperature": 0.3,
        "max_tokens": 1000,
        "system_prompt": """You are an expert evaluator. Compare the actual output to the expected output and provide:
1. A score from 0.0 to 1.0 (where 1.0 is perfect match)
2. A brief reasoning for your score
3. Any specific errors or discrepancies noted

Respond in JSON format:
{
    "score": 0.0-1.0,
    "reasoning": "explanation",
    "errors": ["error1", "error2"] or []
}"""
    }

judge_config = st.session_state.model_configs["default_judge_config"]

col1, col2 = st.columns([1, 2])

with col1:
    judge_config["provider"] = st.selectbox(
        "Judge Model Provider",
        ["openai", "anthropic", "google"],
        index=["openai", "anthropic", "google"].index(judge_config["provider"]),
    )
    
    # Model selection based on provider
    model_options = {
        "openai": ["gpt-4", "gpt-4-turbo-preview", "gpt-3.5-turbo"],
        "anthropic": ["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"],
        "google": ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-1.0-pro"],
    }
    
    current_models = model_options[judge_config["provider"]]
    if judge_config["model"] not in current_models:
        judge_config["model"] = current_models[0]
    
    judge_config["model"] = st.selectbox(
        "Judge Model",
        current_models,
        index=current_models.index(judge_config["model"]),
    )
    
    judge_config["temperature"] = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=judge_config["temperature"],
        step=0.1,
        help="Lower values make output more deterministic",
    )
    
    judge_config["max_tokens"] = st.number_input(
        "Max Tokens",
        min_value=100,
        max_value=4000,
        value=judge_config["max_tokens"],
        step=100,
    )

with col2:
    judge_config["system_prompt"] = st.text_area(
        "Judge System Prompt",
        value=judge_config["system_prompt"],
        height=300,
        help="Instructions for the LLM judge on how to evaluate outputs",
    )

# Save Configuration
st.header("3. Save Configuration")

if st.button("\U0001F4BE Save All Configurations", type="primary", use_container_width=True):
    # Validate that at least one API key is provided
    if not any(st.session_state.api_keys.values()):
        st.error("Please provide at least one API key.")
    else:
        st.success("Configuration saved successfully!")
        st.session_state.model_configs["default_judge_config"] = judge_config
        
        # Show summary
        st.markdown("### Configuration Summary")
        
        # API Keys status
        st.markdown("**API Keys Configured:**")
        for provider, key in st.session_state.api_keys.items():
            if key:
                st.markdown(f"- {provider.title()}: \u2705 Set")
        
        # Model config summary
        st.markdown(f"""
        **Default Judge Configuration:**
        - Provider: {judge_config['provider']}
        - Model: {judge_config['model']}
        - Temperature: {judge_config['temperature']}
        - Max Tokens: {judge_config['max_tokens']}
        """)

# Navigation hint
st.markdown("---")
st.info("\u2705 Once configured, proceed to **Evaluation Setup** to upload data and select scorers.")
