
# firebase_config.py
import firebase_admin
from firebase_admin import credentials, auth, firestore
from pathlib import Path

def initialize_firebase():
    """
    Initialize Firebase Admin SDK if not already initialized.
    Returns the Firebase app instance.s
    """
    try:
        return firebase_admin.get_app()
    except ValueError:
        # Firebase not initialized yet, so initialize it
        cred_path = Path(__file__).parent / 'gem-rush-007-firebase-adminsdk-pzp02-b7a2022e8b.json'
        cred = credentials.Certificate(str(cred_path))
        return firebase_admin.initialize_app(cred)

# Initialize Firebase when the module is imported
app = initialize_firebase()

# Get Firestore client
db = firestore.client()

# Export the services we need
def get_auth():
    return auth

def get_firestore():
    return db

# This allows other modules to just import what they need
__all__ = ['get_auth', 'get_firestore']