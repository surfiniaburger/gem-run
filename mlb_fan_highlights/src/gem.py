import streamlit as st
import streamlit.components.v1 as components
from surfire2 import generate_mlb_analysis
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

def create_secure_iframe(iframe_html: str) -> str:
    """
    Creates a secure iframe HTML string with proper attributes.
    
    Args:
        iframe_html (str): Original iframe HTML
        
    Returns:
        str: Sanitized iframe HTML with security attributes
    """
    if not iframe_html:
        return None
    
    # Remove any existing sandbox attributes
    iframe_html = iframe_html.replace(
        'sandbox="allow-storage-access-by-user-activation allow-scripts allow-same-origin allow-popups allow-popups-to-escape-sandbox"',
        ''
    )
    
    # Add comprehensive sandbox attributes for security
    secure_attributes = (
        'sandbox="allow-scripts allow-same-origin allow-popups '
        'allow-popups-to-escape-sandbox allow-storage-access-by-user-activation"'
    )
    iframe_html = iframe_html.replace('<iframe', f'<iframe {secure_attributes}')
    
    return iframe_html

def main():
    st.set_page_config(
        page_title="MLB Analysis",
        page_icon="⚾",
        layout="wide"
    )

    st.title("MLB Analysis Dashboard ⚾")

    # Sidebar
    with st.sidebar:
        st.markdown("""
        ## About
        This app generates MLB analysis based on your prompt using AI.

        ### How to use
        1. Enter your analysis question
        2. Click 'Generate Analysis'
        3. View both text analysis and visualizations
        
        ### Features
        - Natural language queries
        - Interactive visualizations
        - Real-time MLB data analysis
        """)

    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        prompt = st.text_area(
            "Enter your analysis prompt:",
            "What's the average home team score when the away team scores more than 5 runs?",
            height=100
        )

    with col2:
        st.markdown("<br>" * 2, unsafe_allow_html=True)  # Add some spacing
        analyze_button = st.button(
            "Generate Analysis",
            use_container_width=True,
            type="primary"
        )

    if analyze_button:
        with st.spinner("Generating analysis..."):
            try:
                result = generate_mlb_analysis(prompt)
                
                # Create tabs for organization
                tab1, tab2 = st.tabs(["Analysis", "Visualization"])
                
                with tab1:
                    st.success("Analysis generated successfully!")
                    st.write(result["text"])
                
                with tab2:
                    if result.get("iframe_url"):
                        secure_iframe = create_secure_iframe(result["iframe_url"])
                        if secure_iframe:
                            try:
                                components.html(
                                    secure_iframe,
                                    height=600,
                                    scrolling=True
                                )
                            except Exception as iframe_error:
                                st.warning(
                                    "Could not load visualization. "
                                    "Technical details below:"
                                )
                                with st.expander("Show technical details"):
                                    st.code(secure_iframe, language="html")
                                    st.error(str(iframe_error))
                    else:
                        st.info("No visualization available for this analysis.")
                
            except Exception as e:
                st.error("An error occurred during analysis generation.")
                with st.expander("Show error details"):
                    st.error(str(e))          
if __name__ == "__main__":
    main()