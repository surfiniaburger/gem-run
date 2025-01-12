import streamlit as st
from surfire import generate_mlb_analysis
import streamlit.components.v1 as components

st.title("MLB Analysis")

def sanitize_iframe(iframe_html: str) -> str:
    """
    Sanitize iframe HTML to ensure proper security settings.
    """
    if not iframe_html:
        return None
    
    # Remove any existing sandbox attributes to prevent conflicts
    iframe_html = iframe_html.replace('sandbox="allow-storage-access-by-user-activation allow-scripts allow-same-origin allow-popups allow-popups-to-escape-sandbox"', '')
    
    # Add secure sandbox attributes
    iframe_html = iframe_html.replace('<iframe', '<iframe sandbox="allow-scripts"')
    
    return iframe_html

# Input widget for the prompt
prompt = st.text_input("Enter your analysis prompt:", "What's the average home team score when the away team scores more than 5 runs?")

if st.button("Generate Analysis"):
    try:
        result = generate_mlb_analysis(prompt)
        
        # Check if the result is a tuple (text_response, iframe)
        if isinstance(result, tuple) and len(result) == 2:
            text_response, iframe_html = result
            print(result)
            st.success("Analysis generated successfully!")
            st.write(text_response)
            
            if iframe_html:
                print(iframe_html)
                # Sanitize and render the iframe
                secure_iframe = sanitize_iframe(iframe_html)
                print(secure_iframe)
                if secure_iframe:
                    try:
                        components.html(
                            secure_iframe,
                            height=600,
                            scrolling=True
                        )
                    except Exception as iframe_error:
                        st.warning(f"Could not load visualization: {str(iframe_error)}")
                        st.code(secure_iframe, language="html")
        else:
            # If only text response is returned
            st.success("Analysis generated successfully!")
            st.write(result)
            st.info("No visualization available for this analysis.")
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

st.sidebar.markdown("""
## About
This app generates MLB analysis based on your prompt using AI.

- Enter your analysis question in the text input.
- Click 'Generate Analysis' to see the results.
- Visualizations are displayed when available.
""")