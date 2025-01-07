import os
from google.cloud import storage
from datetime import timedelta

def generate_signed_urls(bucket_name, expiration_days=1):
    """
    Generates signed URLs for all objects in a Google Cloud Storage bucket.

    Args:
        bucket_name (str): The name of the GCS bucket.
        expiration_days (int): The number of days the signed URL should be valid.

    Returns:
        dict: A dictionary mapping object names to signed URLs.
    """
    try:
        # Initialize the GCS client
        storage_client = storage.Client()

        # Get the GCS bucket
        bucket = storage_client.bucket(bucket_name)

        signed_urls = {}
        
        # List all blobs (objects) in the bucket
        blobs = bucket.list_blobs()

        for blob in blobs:
            # Generate a signed URL that is valid for the specified number of days
            expiration_time = timedelta(days=expiration_days)
            signed_url = blob.generate_signed_url(
                version="v4",
                expiration=expiration_time,
                method="GET",
            )
            signed_urls[blob.name] = signed_url
        
        print("Signed URLs generated successfully.")
        return signed_urls
    
    except Exception as e:
        print(f"Error generating signed URLs: {e}")
        return None

if __name__ == "__main__":
    # Input your bucket name and the directory with the images
    bucket_name = "mlb-headshots"  # Replace with your bucket name
    expiration_days = 7 # Set the number of days you want your URL valid
    
    signed_urls = generate_signed_urls(bucket_name, expiration_days)

    if signed_urls:
        # Print the object to url mapping
        print (signed_urls)
        # You will save this mapping as you see fit.