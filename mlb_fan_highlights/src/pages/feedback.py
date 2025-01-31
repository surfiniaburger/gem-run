import streamlit as st
from auth import sign_in_or_sign_up
from user_profile import UserProfile


def feedback_form(user_profile: UserProfile):
    """
    Displays a feedback form and saves the feedback using the provided UserProfile instance.
    
    Args:
        user_profile: An instance of UserProfile for the current user
    """
    st.title("MLB Podcast Generator Feedback 🎙️⚾")

    # Display the generated podcast content here
    st.write("Your generated MLB podcast content goes here...")

    # Feedback form
    st.subheader("We'd love to hear your feedback! 📝")

    # Overall rating
    rating = st.slider(
        "How would you rate this podcast? ⭐",
        min_value=1,
        max_value=5,
        value=3,
        step=1
    )

    # Detailed feedback
    with st.form("feedback_form"):
        st.write("Please provide more detailed feedback:")
        
        content_quality = st.slider("Content Quality 📊", 1, 5, 3)
        audio_quality = st.slider("Audio Quality 🔊", 1, 5, 3)
        
        favorite_segment = st.text_input("What was your favorite segment? 🏆")
        
        improvement = st.text_area("Any suggestions for improvement? 💡")
        
        submitted = st.form_submit_button("Submit Feedback")

    if submitted:
        # Prepare feedback data
        feedback_data = {
            'rating': rating,
            'content_quality': content_quality,
            'audio_quality': audio_quality,
            'favorite_segment': favorite_segment,
            'improvement_suggestions': improvement
        }
        
        # Save feedback using UserProfile
        if user_profile.save_feedback(feedback_data):
            st.success("Thank you for your feedback! ⭐")
        else:
            st.error("Failed to save feedback. Please try again.")



def main():
    # Check if user is authenticated
    if 'user' not in st.session_state:
        st.warning("Please log in to access this page.")
        sign_in_or_sign_up()
        return
    
    # If user is logged in, show the feedback form
    user_profile = UserProfile(st.session_state['user'].uid, st.session_state['user'].email)
    feedback_form(user_profile)

if __name__ == "__main__":
    main()