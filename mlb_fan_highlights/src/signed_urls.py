import os
from google.cloud import storage
from datetime import timedelta
from google.api_core import exceptions
import csv

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

        # Get the bucket
        bucket = storage_client.bucket(bucket_name)
        
        # Test bucket access
        try:
            bucket.reload()
        except exceptions.Forbidden:
            print(f"Permission denied: Service account does not have required permissions on bucket '{bucket_name}'")
            print("Required permissions: storage.buckets.get, storage.objects.list, storage.objects.get")
            return None
        except exceptions.NotFound:
            print(f"Bucket '{bucket_name}' not found")
            return None

        signed_urls = {}
        
        try:
            # List all blobs (objects) in the bucket and convert to list immediately
            blobs = list(bucket.list_blobs())
        except exceptions.Forbidden:
            print("Permission denied: Service account cannot list objects in the bucket")
            print("Required permission: storage.objects.list")
            return None

        if not blobs:
            print("No objects found in the bucket.")
            return {}

        for blob in blobs:
            try:
                # Generate a signed URL that is valid for the specified number of days
                expiration_time = timedelta(days=expiration_days)
                signed_url = blob.generate_signed_url(
                    version="v4",
                    expiration=expiration_time,
                    method="GET"
                )
                signed_urls[blob.name] = signed_url
            except Exception as e:
                print(f"Error generating signed URL for {blob.name}: {str(e)}")
                continue

        if signed_urls:
            print(f"Generated {len(signed_urls)} signed URLs successfully.")
            return signed_urls
        else:
            print("No signed URLs could be generated.")
            return None
    
    except Exception as e:
        print(f"Error generating signed URLs: {str(e)}")
        return None

if __name__ == "__main__":
    bucket_name = "mlb-headshots"  # Your bucket name
    expiration_days = 7
    
    # Verify credentials are set
    key_file_path = "./gem-rush-007-a9765f2ada0e.json" 
    credentials_path = key_file_path
    if not credentials_path:
        print("GOOGLE_APPLICATION_CREDENTIALS environment variable not set")
        exit(1)
    
    print(f"Using credentials from: {credentials_path}")

    
    signed_urls = generate_signed_urls(bucket_name, expiration_days)



    if signed_urls:
        with open("signed_urls.csv", 'w', newline='') as csvfile:
            fieldnames = ["file_name", "signed_url"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for key, value in signed_urls.items():
               writer.writerow({"file_name": key, "signed_url": value})
        print("signed URLs have been saved to signed_urls.csv")
