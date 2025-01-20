
import streamlit as st
import firebase_admin
from firebase_admin import credentials
from firebase_admin import auth
from google.oauth2 import service_account
from google.auth.transport.requests import Request
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
import os
# Load credentials from environment variable
cred = credentials.Certificate(os.environ.get('FIREBASE_SERVICE_ACCOUNT_KEY'))
firebase_admin.initialize_app(cred)

# Google Sign-In Configuration
SCOPES = ['https://www.googleapis.com/auth/userinfo.email', 'https://www.googleapis.com/auth/userinfo.profile']
CLIENT_SECRET_FILE = os.environ.get('GOOGLE_CLIENT_SECRET')

# Function to handle Google Sign-In
def google_sign_in():
    flow = service_account.Credentials.from_service_account_file(
        CLIENT_SECRET_FILE, scopes=SCOPES
    )
    flow.redirect_uri = 'https://mlb-1011675918473.us-central1.run.app/__/auth/handler'  # Your Cloud Run URL
    authorization_url, state = flow.authorization_url(access_type='offline', prompt='consent')
    st.write(f'<a href="{authorization_url}">Sign in with Google</a>', unsafe_allow_html=True)

    # Handle the callback
    if 'code' in st.session_state:
        flow.fetch_token(code=st.session_state.code)
        credentials = flow.credentials
        st.session_state.credentials = credentials

        # Get user information from Google
        service = build('oauth2', 'v2', credentials=credentials)
        user_info = service.userinfo().get().execute()
        st.write(f"Welcome, {user_info['name']}!")

# Streamlit UI
st.title("My Streamlit App")

if 'credentials' not in st.session_state:
    google_sign_in()
else:
    st.write("You are already signed in.")

# ... Rest of your Streamlit app logic ...