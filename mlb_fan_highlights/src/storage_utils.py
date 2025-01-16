from google.cloud import storage
from datetime import timedelta
from google.api_core.exceptions import NotFound

# Constants for Google Cloud Storage
GCS_BUCKET_NAME = "mlb-podcast-bucket"
GCS_LOCATION = "US"
GCS_PROJECT = "gem-rush-007"

def create_gcs_bucket(bucket_name, location):
    """Creates a Google Cloud Storage bucket if it doesn't exist."""
    storage_client = storage.Client(project=GCS_PROJECT)
    bucket = storage_client.bucket(bucket_name)
    
    try:
        storage_client.get_bucket(bucket_name)
        print(f"Bucket with name : {bucket_name} already exist")
    except NotFound:
        print(f"Creating bucket with name: {bucket_name}")
        bucket = storage_client.create_bucket(bucket, location=location)
        print(f"Bucket {bucket} created in {location}")
    except Exception as e:
        raise Exception(f"An error has occured while creating gcs bucket, : {e}")

def upload_audio_to_gcs(audio_content: bytes, file_name: str) -> str:
    """Uploads audio to GCS and returns a signed URL."""
    try:
        # Create a google cloud client
        storage_client = storage.Client(project=GCS_PROJECT)
        # Create or get the bucket
        create_gcs_bucket(GCS_BUCKET_NAME, GCS_LOCATION)
        bucket = storage_client.bucket(GCS_BUCKET_NAME)
        # Upload the file
        blob = bucket.blob(file_name)
        blob.upload_from_string(audio_content, content_type="audio/mp3")
        #Generate the signed URL
        url = blob.generate_signed_url(
            version="v4",
            expiration=timedelta(minutes=15),
            method="GET"
        )
        return url
    except Exception as e:
        raise Exception(f"An error occurred while uploading audio to GCS: {e}")