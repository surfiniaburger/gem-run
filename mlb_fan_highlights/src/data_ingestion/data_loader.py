# mlb_fan_highlights/src/data_ingestion/data_loader.py

from google.cloud import storage


def load_data(bucket_name, file_pattern):
    """Loads data from a Cloud Storage bucket based on a file pattern."""

    client = storage.Client()
    bucket = client.bucket(bucket_name)

    data = []
    for blob in bucket.list_blobs(prefix=file_pattern):
        data_string = blob.download_as_string()
        data.append(json.loads(data_string))

    return data


if __name__ == "__main__":
    # Example usage
    processed_data = load_data(gcs_bucket, "gumbo/processed/*.json")
    print(f"Loaded {len(processed_data)} records from Cloud Storage.")