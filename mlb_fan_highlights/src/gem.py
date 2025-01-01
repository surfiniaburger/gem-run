import streamlit as st
from surfire import generate_mlb_analysis 
# Assuming the generate_mlb_analysis function is imported or defined here

st.title("MLB Analysis")

# Input widget for the prompt
prompt = st.text_input("Enter your analysis prompt:", "What's the average home team score when the away team scores more than 5 runs?")

if st.button("Generate Analysis"):
    try:
        analysis_result = generate_mlb_analysis(prompt)
        st.success("Analysis generated successfully!")
        st.write(analysis_result)
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

st.sidebar.markdown("""
## About
This app generates MLB analysis based on your prompt using AI.

- Enter your analysis question in the text input.
- Click 'Generate Analysis' to see the results.
""")