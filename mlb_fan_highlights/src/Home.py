# pages/analytics_page.py
import streamlit as st
from firebase_config import get_auth, get_firestore
from datetime import datetime
from firebase_admin import firestore
import uuid
import pytz
from user_profile import UserProfile

# Get Firebase services
auth = get_auth()
db = get_firestore()


# Add Google Analytics tracking code
ga_script = """
<!-- Google tag (gtag.js) -->
<script async src="https://www.googletagmanager.com/gtag/js?id=G-98KGSC9LXG"></script>
<script>
    window.dataLayer = window.dataLayer || [];
    function gtag(){dataLayer.push(arguments);}
    gtag('js', new Date());
    gtag('config', 'G-98KGSC9LXG');
</script>
"""
# Inject GA script using streamlit HTML function
def inject_ga():
    st.components.v1.html(ga_script, height=0)


# Inject GA script into Streamlit
st.set_page_config(
    page_title="MLB Podcast Generator",
    layout="wide",
    initial_sidebar_state="expanded"
)


def create_analytics_landing():
    # Custom CSS for animations and styling
    st.markdown("""
        <style>
        @keyframes float {
            0% { transform: translateY(0px); }
            50% { transform: translateY(-10px); }
            100% { transform: translateY(0px); }
        }
        
        @keyframes slide-in {
            0% { transform: translateX(-100%); opacity: 0; }
            100% { transform: translateX(0); opacity: 1; }
        }
        
        .stat-container {
            display: flex;
            justify-content: space-around;
            flex-wrap: wrap;
            margin: 30px 0;
            animation: slide-in 1s ease-out;
        }
        
        .stat-box {
            background: white;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
            margin: 10px;
            text-align: center;
            min-width: 200px;
            transition: transform 0.3s ease;
        }
        
        .stat-box:hover {
            transform: translateY(-5px);
        }
        
        .animated-header {
            text-align: center;
            padding: 30px;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: white;
            border-radius: 15px;
            margin-bottom: 30px;
            animation: slide-in 0.8s ease-out;
        }
        
        .floating-emoji {
            font-size: 2em;
            animation: float 3s ease-in-out infinite;
            display: inline-block;
            margin: 0 10px;
        }
        
        .insights-container {
            padding: 20px;
            background: rgba(255, 255, 255, 0.9);
            border-radius: 15px;
            margin: 20px 0;
            animation: slide-in 1.2s ease-out;
        }
        
        .welcome-banner {
            background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 30px;
            text-align: center;
            animation: slide-in 0.5s ease-out;
        }
        </style>
    """, unsafe_allow_html=True)

    # Welcome Banner
    st.markdown("""
        <div class="welcome-banner">
            <h2>MLB Analytics Dashboard</h2>
            <p>Dive into your baseball insights and statistics</p>
        </div>
    """, unsafe_allow_html=True)

    # Animated Header with Floating Emojis
    st.markdown("""
        <div class="animated-header">
            <span class="floating-emoji" style="animation-delay: 0s">📊</span>
            <span class="floating-emoji" style="animation-delay: 0.5s">⚾</span>
            <span class="floating-emoji" style="animation-delay: 1s">📈</span>
            <h1>Your Baseball Analytics Hub</h1>
        </div>
    """, unsafe_allow_html=True)

    # Stats Overview
    st.markdown("""
        <div class="stat-container">
            <div class="stat-box">
                <h3>Team Performance</h3>
                <p>Track your favorite team's progress</p>
                <span style="font-size: 2em;">📈</span>
            </div>
            <div class="stat-box">
                <h3>Player Stats</h3>
                <p>Detailed player analytics</p>
                <span style="font-size: 2em;">👤</span>
            </div>
            <div class="stat-box">
                <h3>Game Analysis</h3>
                <p>In-depth game breakdowns</p>
                <span style="font-size: 2em;">🎯</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Insights Section
    st.markdown("""
        <div class="insights-container">
            <h2 style="color: #1e3c72;">Featured Insights</h2>
            <ul style="list-style-type: none; padding: 0;">
                <li style="margin: 10px 0;">🏆 Season Highlights</li>
                <li style="margin: 10px 0;">📊 Performance Metrics</li>
                <li style="margin: 10px 0;">🎤 Podcast Analytics</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

def handle_authentication(email, password, auth_type):
 """Enhanced authentication handler with detailed error handling"""
 try:
     if auth_type == "Sign In":
         user = auth.get_user_by_email(email)
         auth_user = auth.get_user(user.uid)
         st.session_state['user'] = auth_user
         
         # Create/update user profile
         profile = UserProfile(user.uid, email)
         profile.create_or_update()
         
         st.success(f"Welcome back, {email}!")
         return True
         
     else:  # Sign Up
         # Password validation
         if len(password) < 6:
             st.error("Password must be at least 6 characters long")
             return False
             
         user = auth.create_user(email=email, password=password)
         auth_user = auth.get_user(user.uid)
         st.session_state['user'] = auth_user
         
         # Create new user profile
         profile = UserProfile(user.uid, email)
         profile.create_or_update({
             'account_type': 'free',
             'podcasts_generated': 0
         })
         
         st.success(f"Welcome to MLB Podcast Generator, {email}!")
         return True
         
 except auth.EmailAlreadyExistsError:
     st.error("This email is already registered. Please sign in instead.")
 except auth.UserNotFoundError:
     st.error("No account found with this email. Please sign up.")
 except auth.InvalidEmailError:
     st.error("Please enter a valid email address.")
 except auth.WeakPasswordError:
     st.error("Password is too weak. Please choose a stronger password.")
 except Exception as e:
     st.error(f"Authentication error: {str(e)}")
 return False



def sign_in_or_sign_up():
 """Enhanced sign in/sign up form with validation"""
 auth_type = st.radio("Sign In or Sign Up", ["Sign In", "Sign Up"])
 
 with st.form(key='auth_form'):
     email = st.text_input("Email")
     password = st.text_input("Password", type="password")
     submit_button = st.form_submit_button(auth_type)
     
     if submit_button:
         if not email or not password:
             st.error("Please fill in all fields.")
             return
         
         if handle_authentication(email, password, auth_type):
             # Use rerun() to refresh the page after successful authentication
             st.rerun()



def main():
    # Inject the GA script
    inject_ga()

    # Check if user is in session
    if 'user' not in st.session_state:
        st.warning("Please log in to access this page.")
        sign_in_or_sign_up()
        return
    
    # If user is logged in, show the analytics landing page
    create_analytics_landing()

    # Get user profile and display relevant information
    if 'user' in st.session_state:
        profile = UserProfile(st.session_state['user'].uid, st.session_state['user'].email)
        user_data = profile.get_profile()
        
        if user_data:
            # Display user-specific analytics in a styled container
            st.markdown("""
                <div style="background: white; padding: 20px; border-radius: 10px; margin-top: 20px;">
                    <h3 style="color: #1e3c72;">Your Activity Summary</h3>
                """, unsafe_allow_html=True)
            
            # Display usage statistics with proper datetime handling
            usage_stats = profile.get_usage_stats()
            if usage_stats:
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Podcasts Generated", usage_stats['podcasts_generated'])
                with col2:
                    if usage_stats['account_created']:
                        # Convert account_created to UTC if it's naive
                        account_created = usage_stats['account_created']
                        if account_created.tzinfo is None:
                            account_created = pytz.UTC.localize(account_created)
                        
                        # Use UTC for current time as well
                        current_time = datetime.now(pytz.UTC)
                        days_active = (current_time - account_created).days
                        st.metric("Days as Member", days_active)
    # Show podcast history
    history = profile.get_podcast_history()
    if history:
         st.expander("Your Previous Podcasts")
         for podcast in history:
             st.audio(podcast['url'])
             st.caption(f"Generated: {podcast['generated_at']}")
                        
if __name__ == "__main__":
    main()