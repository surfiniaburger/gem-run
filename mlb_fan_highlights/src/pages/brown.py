import streamlit as st
from google.cloud import storage, secretmanager_v1
from google.cloud import aiplatform
import vertexai
from vertexai.vision_models import Image, MultiModalEmbeddingModel
from langchain_google_alloydb_pg import AlloyDBEngine, AlloyDBVectorStore, Column
from datetime import datetime, timedelta
import asyncio
import logging
from typing import Optional
import sys
import os
from firebase_admin import firestore
from firebase_config import get_auth, get_firestore
import uuid
from sqlalchemy import text

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MLBApp")

def get_secret(project_id: str, secret_id: str, version_id: str = 'latest'):
    client = secretmanager_v1.SecretManagerServiceClient()
    name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"
    response = client.access_secret_version(request={"name": name})
    return response.payload.data.decode('UTF-8')

async def test_alloydb_connection(project_id: str, region: str, cluster: str, 
                                instance: str, database: str, db_user: str, 
                                db_password: str) -> bool:
    try:
        engine = await AlloyDBEngine.afrom_instance(
            project_id=project_id,
            region=region,
            cluster=cluster,
            instance=instance,
            database=database,
            user=db_user,
            password=db_password
        )
        
        async with engine._pool.connect() as conn:
            result = await conn.execute(text("SELECT 1"))
            await result.fetchone()
            return True
    except Exception as e:
        logger.error(f"Database connection test failed: {str(e)}")
        return False

def initialize_vertexai(project_id: str, region: str):
    vertexai.init(project=project_id, location=region)
    return MultiModalEmbeddingModel.from_pretrained("multimodalembedding@001")

class MLBApp:
    def __init__(self):
        self.auth = get_auth()
        self.db = get_firestore()
        self.project_id = "gem-rush-007"
        self.region = "us-east4"
        self.bucket_name = "mlb-headshot"
        
    async def test_database_setup(self):
        try:
            db_password = get_secret(self.project_id, "ALLOYDB_PASSWORD")
            connection_successful = await test_alloydb_connection(
                project_id=self.project_id,
                region=self.region,
                cluster="my-cluster",
                instance="my-cluster-primary",
                database="player_headshots",
                db_user='postgres',
                db_password=db_password
            )
            return connection_successful
        except Exception as e:
            logger.error(f"Database setup test failed: {str(e)}")
            return False

    def render_ui(self):
        st.title("MLB App with AlloyDB Integration")
        
        if 'user' not in st.session_state:
            self.sign_in_or_sign_up()
            return
            
        if st.sidebar.button("Test AlloyDB Connection"):
            with st.spinner("Testing database connection..."):
                if asyncio.run(self.test_database_setup()):
                    st.success("Successfully connected to AlloyDB!")
                else:
                    st.error("Failed to connect to AlloyDB. Check logs for details.")
        
        # Rest of your existing UI code here
        self.render_main_content()

    def sign_in_or_sign_up(self):
        auth_type = st.radio("Sign In or Sign Up", ["Sign In", "Sign Up"])
        
        with st.form(key='auth_form'):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            submit_button = st.form_submit_button(auth_type)
            
            if submit_button and email and password:
                if self.handle_authentication(email, password, auth_type):
                    st.rerun()

    def handle_authentication(self, email, password, auth_type):
        try:
            if auth_type == "Sign In":
                user = self.auth.get_user_by_email(email)
                st.session_state['user'] = self.auth.get_user(user.uid)
                self.create_or_update_user_profile(user.uid, email)
                return True
            else:
                if len(password) < 6:
                    st.error("Password must be at least 6 characters long")
                    return False
                    
                user = self.auth.create_user(email=email, password=password)
                st.session_state['user'] = self.auth.get_user(user.uid)
                self.create_or_update_user_profile(user.uid, email, is_new=True)
                return True
                
        except Exception as e:
            st.error(f"Authentication error: {str(e)}")
            return False

    def create_or_update_user_profile(self, uid, email, is_new=False):
        try:
            user_ref = self.db.collection('users').document(uid)
            data = {
                'email': email,
                'last_login': datetime.now()
            }
            
            if is_new:
                data.update({
                    'account_created': datetime.now(),
                    'account_type': 'free',
                    'podcasts_generated': 0
                })
            
            user_ref.set(data, merge=True)
            
        except Exception as e:
            logger.error(f"Error updating user profile: {str(e)}")

    def render_main_content(self):
        # Add your existing MLB app content here
        st.write("Main content area - Add your MLB-specific features here")

def main():
    app = MLBApp()
    app.render_ui()

if __name__ == "__main__":
    main()