from google.cloud import logging as cloud_logging
from google.cloud import secretmanager
from google.cloud import storage
from google.oauth2 import service_account
from datetime import timedelta
import json

class GCSHandler:
    def __init__(self, secret_id: str, secret_version: str = 'latest'):
        """
        Initialize GCS Handler with logging and Secret Manager integration.
        
        Args:
            secret_id: ID of the secret containing service account key
            secret_version: Version of the secret (default is 'latest')
        """
        # Configure cloud logging
        self.logger = cloud_logging.Client().logger('gcs-handler')
        
        # Project and bucket details
        self.project_id = "gem-rush-007"
        self.bucket_name = "mlb-podcast-bucket"
        
        try:
            # Initialize Secret Manager client
            secret_client = secretmanager.SecretManagerServiceClient()
            
            # Retrieve secret
            secret_path = secret_client.secret_version_path(
                self.project_id, secret_id, secret_version
            )
            
            # Log secret retrieval attempt
            self.logger.log_text(f"Attempting to retrieve secret: {secret_id}")
            
            # Access the secret payload
            response = secret_client.access_secret_version(request={"name": secret_path})
            service_account_key = json.loads(response.payload.data.decode('UTF-8'))
            
            # Create credentials
            self.credentials = service_account.Credentials.from_service_account_info(
                service_account_key,
                scopes=['https://www.googleapis.com/auth/cloud-platform']
            )
            
            # Initialize storage client
            self.storage_client = storage.Client(
                project=self.project_id,
                credentials=self.credentials
            )
            
            self.logger.log_text("Successfully initialized GCS Handler")
        
        except Exception as e:
            # Log initialization error
            error_message = f"Initialization error: {str(e)}"
            self.logger.log_text(error_message, severity='ERROR')
            raise

    def upload_audio(self, audio_content: bytes, file_name: str) -> str:
        """
        Upload audio to GCS and return a signed URL with logging.
        
        Args:
            audio_content: Audio file content in bytes
            file_name: Name for the uploaded file
        
        Returns:
            str: Signed URL for accessing the uploaded file
        """
        try:
            # Log upload attempt
            self.logger.log_text(f"Attempting to upload file: {file_name}")
            
            # Get the bucket
            bucket = self.storage_client.bucket(self.bucket_name)
            
            # Create the blob and upload the file
            blob = bucket.blob(file_name)
            blob.upload_from_string(audio_content, content_type="audio/mp3")
            
            # Generate signed URL
            url = blob.generate_signed_url(
                version="v4",
                expiration=timedelta(minutes=10),
                method="GET",
                service_account_email=self.credentials.service_account_email,
                access_token=None,
                credentials=self.credentials
            )
            
            # Log successful upload
            self.logger.log_text(f"Successfully uploaded {file_name}")
            
            return url
        
        except Exception as e:
            # Log upload error
            error_message = f"Upload error for {file_name}: {str(e)}"
            self.logger.log_text(error_message, severity='ERROR')
            raise

    def verify_bucket_access(self) -> bool:
        """
        Verify bucket access with logging.
        
        Returns:
            bool: True if bucket access is successful
        """
        try:
            # Log access verification attempt
            self.logger.log_text(f"Verifying access to bucket: {self.bucket_name}")
            
            bucket = self.storage_client.bucket(self.bucket_name)
            # Try to get bucket metadata
            bucket.reload()
            
            # Log successful access
            self.logger.log_text(f"Successfully verified access to {self.bucket_name}")
            return True
        
        except Exception as e:
            # Log access verification error
            error_message = f"Bucket access error: {str(e)}"
            self.logger.log_text(error_message, severity='ERROR')
            raise