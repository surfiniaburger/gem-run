from google.cloud import logging as cloud_logging
from google.cloud import secretmanager
from google.cloud import storage
from google.oauth2 import service_account
from datetime import timedelta, datetime, UTC
import json
import urllib.parse
import base64

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
                expiration=timedelta(minutes=3600),
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



    def refresh_signed_url(self, expired_url: str) -> str:
        """
        Refresh an expired signed URL for a GCS object.
        
        Args:
            expired_url: The expired signed URL
            
        Returns:
            str: A new signed URL for the same object
            
        Raises:
            ValueError: If the URL is not a valid GCS signed URL
        """
        try:
            # Parse the URL to extract the blob path
            parsed_url = urllib.parse.urlparse(expired_url)
            path_parts = parsed_url.path.split('/')
            
            # The blob name should be everything after the bucket name in the path
            if self.bucket_name not in path_parts:
                raise ValueError("Invalid GCS URL: bucket name not found")
            
            bucket_index = path_parts.index(self.bucket_name)
            blob_name = '/'.join(path_parts[bucket_index + 1:])
            
            # Log refresh attempt
            self.logger.log_text(f"Attempting to refresh signed URL for blob: {blob_name}")
            
            # Get the blob and generate a new signed URL
            bucket = self.storage_client.bucket(self.bucket_name)
            blob = bucket.blob(blob_name)
            
            new_url = blob.generate_signed_url(
                version="v4",
                expiration=timedelta(minutes=3600),
                method="GET",
                service_account_email=self.credentials.service_account_email,
                access_token=None,
                credentials=self.credentials
            )
            
            # Log successful refresh
            self.logger.log_text(f"Successfully refreshed signed URL for {blob_name}")
            
            return new_url
            
        except Exception as e:
            # Log refresh error
            error_message = f"URL refresh error: {str(e)}"
            self.logger.log_text(error_message, severity='ERROR')
            raise

    def is_url_expired(self, signed_url: str) -> bool:
        """
        Check if a signed URL has expired.
        
        Args:
            signed_url: The signed URL to check
            
        Returns:
            bool: True if the URL has expired, False otherwise
        """
        try:
            # Parse the URL to extract the expiration timestamp
            parsed_url = urllib.parse.urlparse(signed_url)
            query_params = urllib.parse.parse_qs(parsed_url.query)
            
            # Extract expiration from X-Goog-Expires parameter
            if 'X-Goog-Expires' in query_params:
                expires = int(query_params['X-Goog-Expires'][0])
                # Extract timestamp from X-Goog-Date parameter
                if 'X-Goog-Date' in query_params:
                    date_str = query_params['X-Goog-Date'][0]
                    start_time = datetime.strptime(date_str, '%Y%m%dT%H%M%SZ')
                    expiration_time = start_time + timedelta(seconds=expires)
                    
                    # Compare with current time
                    return datetime.now(UTC) > expiration_time
            
            # If we can't parse the expiration, assume it's expired to be safe
            return True
            
        except Exception as e:
            # Log error and assume expired to be safe
            error_message = f"Error checking URL expiration: {str(e)}"
            self.logger.log_text(error_message, severity='ERROR')
            return True

    def upload_image(self, image_content: bytes, file_name: str) -> str:
      """Upload image to GCS and return signed URL.
    
    Args:
        image_content: Image bytes
        file_name: Name for the uploaded file
    
    Returns:
        str: Signed URL for accessing the uploaded image
    """
      try:
        self.logger.log_text(f"Uploading image: {file_name}")
        bucket = self.storage_client.bucket(self.bucket_name)
        blob = bucket.blob(file_name)
        blob.upload_from_string(image_content, content_type="image/png")
        
        url = blob.generate_signed_url(
            version="v4",
            expiration=timedelta(minutes=3600),
            method="GET",
            service_account_email=self.credentials.service_account_email,
            access_token=None,
            credentials=self.credentials
        )
        
        return url
        
      except Exception as e:
        self.logger.log_text(f"Image upload error for {file_name}: {str(e)}", severity='ERROR')
        raise
      

    def signed_url_to_gcs_uri(self, signed_url: str) -> str:
      """Convert a signed URL back to GCS URI format.
    
    Args:
        signed_url: The signed URL to convert
        
    Returns:
        str: GCS URI (gs://{bucket}/{blob})
    """
      try:
        parsed_url = urllib.parse.urlparse(signed_url)
        
        # Handle direct GCS URIs
        if parsed_url.scheme == "gs":
            return signed_url
            
        # Validate URL structure
        if not parsed_url.netloc.endswith("storage.googleapis.com"):
            raise ValueError("Invalid GCS URL format")
            
        path_parts = parsed_url.path.lstrip('/').split('/')
        
        if len(path_parts) < 2:
            raise ValueError("Invalid GCS URL path structure")
            
        # Handle URL-encoded paths
        decoded_path = urllib.parse.unquote('/'.join(path_parts))
        return f"gs://{decoded_path}"
        
      except Exception as e:
        self.logger.log_text(f"GCS URI conversion failed: {str(e)}", severity='ERROR')
        raise