"""Session state helpers."""

import streamlit as st
from typing import Any


def get_state(key: str, default: Any = None) -> Any:
    """Retrieve a value from session state."""
    return st.session_state.get(key, default)


def set_state(key: str, value: Any) -> None:
    """Set a value in session state."""
    st.session_state[key] = value
