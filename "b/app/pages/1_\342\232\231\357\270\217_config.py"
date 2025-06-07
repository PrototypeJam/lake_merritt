import streamlit as st
from app import state

st.title("Config")
api_key = st.text_input("OpenAI API Key")
if api_key:
    state.set_state("api_key", api_key)
