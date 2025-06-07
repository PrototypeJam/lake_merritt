"""Streamlit entrypoint."""

from core import logging_config  # noqa: F401
from app import pages  # type: ignore  # ensure package import


if __name__ == "__main__":
    import streamlit as st

    st.switch_page("app/pages/1_config.py")
