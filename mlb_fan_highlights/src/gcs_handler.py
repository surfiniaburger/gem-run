from google.cloud import storage
from google.oauth2 import service_account
from datetime import timedelta, datetime
from google.api_core.exceptions import NotFound
import os
import json

class GCSHandler:
    def __init__(self, key_file_path: str):
        """
        Initialize GCS Handler with explicit key file path.
        
        Args:
            key_file_path: Path to the service account JSON key file
        """
        if not os.path.exists(key_file_path):
            raise FileNotFoundError(f"Key file not found at: {key_file_path}")
            
        self.key_file_path = key_file_path
        self.bucket_name = "mlb-podcast-bucket"
        self.project_id = "gem-rush-007"
        
        # Load credentials directly from key file
        self.credentials = service_account.Credentials.from_service_account_file(
            self.key_file_path,
            scopes=['https://www.googleapis.com/auth/cloud-platform']
        )
        
        self.storage_client = storage.Client(
            project=self.project_id,
            credentials=self.credentials
        )

    def upload_audio(self, audio_content: bytes, file_name: str) -> str:
        """
        Upload audio to GCS and return a signed URL.
        
        Args:
            audio_content: Audio file content in bytes
            file_name: Name for the uploaded file
            
        Returns:
            str: Signed URL for accessing the uploaded file
        """
        try:
            # Get the bucket
            bucket = self.storage_client.bucket(self.bucket_name)
            
            # Create the blob and upload the file
            blob = bucket.blob(file_name)
            blob.upload_from_string(audio_content, content_type="audio/mp3")
            
            # Generate signed URL using the same parameters as the command line
            url = blob.generate_signed_url(
                version="v4",
                expiration=timedelta(minutes=10),  # Matching the 10m from command line
                method="GET",
                service_account_email=self.credentials.service_account_email,
                access_token=None,  # Force usage of private key
                credentials=self.credentials
            )
            
            return url
            
        except Exception as e:
            raise Exception(f"Error uploading audio to GCS: {str(e)}")

    def verify_bucket_access(self) -> bool:
        """
        Verify that we can access the bucket with current credentials.
        Returns True if successful, raises exception if not.
        """
        try:
            bucket = self.storage_client.bucket(self.bucket_name)
            # Try to get bucket metadata
            bucket.reload()
            return True
        except Exception as e:
            raise Exception(f"Cannot access bucket {self.bucket_name}: {str(e)}")