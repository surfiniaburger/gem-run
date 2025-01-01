# pages/iframe_page.py
import streamlit as st

st.title("Looker Studio Report")

# Add the iframe to this page
st.components.v1.iframe(
    src="https://lookerstudio.google.com/embed/reporting/57ebdcdb-9526-44d3-9e47-4d01994f6f1c/page/eiCbE",
    width=600,
    height=450,
    scrolling=True
)