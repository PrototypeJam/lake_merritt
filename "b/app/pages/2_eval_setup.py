import streamlit as st

st.title("Evaluation Setup")
upload = st.file_uploader("CSV File")
if upload:
    st.success("File uploaded")
