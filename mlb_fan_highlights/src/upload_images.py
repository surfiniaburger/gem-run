import os
from google.cloud import storage
from google.api_core.exceptions import NotFound

def upload_image_to_gcs(bucket_name, source_file_path, destination_blob_name):
    """Uploads an image to Google Cloud Storage and makes it publicly accessible."""

    try:
        # Initialize the GCS client
        storage_client = storage.Client()
        
        # Get the GCS bucket
        bucket = storage_client.bucket(bucket_name)

        # Check if the bucket exists, create if it doesn't
        try:
            bucket.exists()
        except NotFound:
            print(f"Bucket '{bucket_name}' not found. Creating it...")
            bucket = storage_client.create_bucket(bucket_name)
            print(f"Bucket '{bucket_name}' created.")
        
        # Get a reference to the blob
        blob = bucket.blob(destination_blob_name)

        # Upload the file to GCS
        print(f"Uploading '{source_file_path}' to '{destination_blob_name}'...")
        blob.upload_from_filename(source_file_path)

        # Make the file public
        blob.make_public()
        print(f"File '{destination_blob_name}' is now publicly accessible.")
        
        # Construct public url
        public_url = f"https://storage.googleapis.com/{bucket_name}/{destination_blob_name}"
        print(f"The public url is: {public_url}")
        return public_url
    
    except NotFound as e:
        print(f"Error: The bucket '{bucket_name}' could not be found or created. Error details: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during upload: {e}")
    return None
    

def upload_images_from_directory(bucket_name, source_dir):
    """
    Uploads all images from a local directory to a GCS bucket.

    Args:
        bucket_name (str): The name of the GCS bucket.
        source_dir (str): Path to the directory containing the images.
    """
    for filename in os.listdir(source_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
           source_file_path = os.path.join(source_dir, filename)
           public_url = upload_image_to_gcs(bucket_name, source_file_path, filename)
           if public_url:
              print (f"Uploaded '{filename}' to bucket '{bucket_name}' and made it public accessible at: {public_url}")


if __name__ == "__main__":
    # Input your bucket name and the directory with the images
    bucket_name = "mlb-headshots"
    source_directory = "../../datasets/digital-diamond-dugout"

    if not os.path.isdir(source_directory):
       print(f"Error: the image directory does not exist '{source_directory}'.")
    else:
       upload_images_from_directory(bucket_name, source_directory)



