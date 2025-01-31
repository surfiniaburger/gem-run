# auth.py
import streamlit as st
from firebase_config import get_auth
from user_profile import UserProfile

auth = get_auth()

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