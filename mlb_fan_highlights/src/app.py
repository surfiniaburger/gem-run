# pages/iframe_page.py
import streamlit as st

def main():
    # Check if user is in session
    if 'user' not in st.session_state:
        st.warning("Please log in to access this page.")
        # Import the sign_in_or_sign_up function from the main script
        from cloud import sign_in_or_sign_up
        sign_in_or_sign_up()
        return

    st.title("Looker Studio Report")
    # Add the iframe to this page
    st.components.v1.iframe(
        src="https://lookerstudio.google.com/embed/reporting/57ebdcdb-9526-44d3-9e47-4d01994f6f1c/page/eiCbE",
        width=600,
        height=450,
        scrolling=True
    )

if __name__ == "__main__":
    main()