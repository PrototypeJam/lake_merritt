"""
Page 1: System & Model Configuration
"""

import asyncio
import os
from typing import Any, Dict

import streamlit as st
from dotenv import load_dotenv

from services.llm_clients import create_llm_client

# Load environment variables
load_dotenv()

st.title("\u2699\ufe0f System & Model Configuration")
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
}""",
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

if st.button(
    "\U0001f4be Save & Validate All Configurations",
    type="primary",
    use_container_width=True,
):

    async def validate_key(provider: str, key: str) -> bool:
        """Test a single API key with a lightweight model call.

        Args:
            provider: Name of the provider.
            key: API key to validate.

        Returns:
            True if the key is valid, otherwise False.
        """

        st.session_state.validation_results[provider] = {"status": "pending"}
        try:
            test_models: Dict[str, str] = {
                "openai": "gpt-3.5-turbo",
                "anthropic": "claude-3-haiku-20240307",
                "google": "gemini-1.5-flash",
            }
            test_prompt = "Generate a single, short, safe-for-work sentence about space exploration."
            messages = [{"role": "user", "content": test_prompt}]

            client = create_llm_client(provider, key)
            response = await client.generate(
                messages,
                model=test_models.get(provider),
                temperature=0.7,
                max_tokens=100,
            )

            st.session_state.validation_results[provider] = {
                "status": "success",
                "response": response,
            }
            return True
        except Exception as e:  # noqa: BLE001
            st.session_state.validation_results[provider] = {
                "status": "failure",
                "error": str(e),
            }
            if provider in st.session_state.api_keys:
                st.session_state.api_keys[provider] = ""
            return False

    if "validation_results" not in st.session_state:
        st.session_state.validation_results = {}

    async def run_all_validations() -> None:
        """Run validation for each provided API key concurrently."""

        tasks = []
        for provider, key in st.session_state.api_keys.items():
            if key:
                tasks.append(validate_key(provider, key))
        await asyncio.gather(*tasks)

    asyncio.run(run_all_validations())

    all_valid = True
    for provider, result in st.session_state.validation_results.items():
        if result.get("status") == "success":
            st.success(f"✅ {provider.title()} API key is valid and working!")
            with st.expander(f"Test response from {provider.title()}", expanded=False):
                st.write(result["response"])
        elif result.get("status") == "failure":
            st.error(
                f"❌ {provider.title()} API key validation failed: {result['error']}"
            )
            all_valid = False

    if all_valid and any(st.session_state.api_keys.values()):
        st.session_state.model_configs["default_judge_config"] = judge_config
        st.success("All configurations saved successfully.")
    elif not any(st.session_state.api_keys.values()):
        st.error("Please provide at least one API key.")
    else:
        st.warning("Configuration not saved. Please fix invalid keys and try again.")

# Navigation hint
st.markdown("---")
st.info(
    "\u2705 Once configured, proceed to **Evaluation Setup** to upload data and select scorers."
)
