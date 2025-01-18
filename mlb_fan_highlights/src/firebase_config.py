# firebase_config.py
import firebase_admin
from firebase_admin import credentials, auth, firestore
from pathlib import Path
import os

def initialize_firebase():
    """
    Initialize Firebase Admin SDK if not already initialized.
    Returns the Firebase app instance.
    
    In Cloud Run: Uses default credentials from service account
    Local dev: Uses service account JSON file
    """
    try:
        return firebase_admin.get_app()
    except ValueError:
        # Check if running in Cloud Run
        is_cloud_run = os.getenv('K_SERVICE')
        
        if is_cloud_run:
            # In Cloud Run: use default credentials
            return firebase_admin.initialize_app()
        else:
            # Local development: use service account file
            cred_path = Path(__file__).parent / 'gem-rush-007-firebase-adminsdk-pzp02-b7a2022e8b.json'
            if not cred_path.exists():
                raise FileNotFoundError(
                    "Firebase credentials file not found. "
                    "For local development, place your service account JSON file in the same directory as this script. "
                    "In Cloud Run, this file is not needed as the service account credentials are used automatically."
                )
            cred = credentials.Certificate(str(cred_path))
            return firebase_admin.initialize_app(cred)

# Initialize Firebase when the module is imported
try:
    app = initialize_firebase()
except Exception as e:
    print(f"Warning: Firebase initialization failed: {e}")
    # You might want to handle this differently depending on your needs
    app = None

# Get Firestore client
db = firestore.client() if app else None

def get_auth():
    """Get Firebase Auth instance"""
    if not app:
        raise RuntimeError("Firebase not properly initialized")
    return auth

def get_firestore():
    """Get Firestore instance"""
    if not db:
        raise RuntimeError("Firestore not properly initialized")
    return db

# This allows other modules to just import what they need
__all__ = ['get_auth', 'get_firestore']