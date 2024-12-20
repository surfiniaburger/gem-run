# mlb_fan_highlights/src/data_ingestion/gumbo_subscriber.py

import json
from google.cloud import pubsub_v1
from google.cloud import storage 

# Replace with your project ID and Pub/Sub topic name
project_id = "gem-creation"
venueId= ""
topic_name = f"projects/{project_id}/topics/mlbam.feed.2.0.game.state.lite.{venueId}"


def subscribe_to_gumbo_data(storage_client, bucket_name, file_prefix="gumbo/raw/"):
    """
    Continuously subscribes to the Gumbo data Pub/Sub topic and processes messages,
    saving raw data to Cloud Storage.

    Args:
        storage_client (google.cloud.storage.Client): A Google Cloud Storage client.
        bucket_name (str): The name of the Cloud Storage bucket to store raw data.
        file_prefix (str, optional): The prefix for Cloud Storage filenames. Defaults to "gumbo/raw/".
    """

    subscriber = pubsub_v1.SubscriberClient()
    subscription_path = subscriber.topic_path(project_id, topic_name)

    def callback(message):
        """
        Callback function triggered for each received message.

        Args:
            message (pubsub_v1.message.Message): The Pub/Sub message containing Gumbo data.
        """

        print(f"Received message: {message.data}")

        try:
            # Parse the JSON data from the message
            data = json.loads(message.data)

            # Generate a unique filename for each message
            filename = f"{file_prefix}{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"

            # Upload the Gumbo data to Cloud Storage
            blob = storage_client.bucket(bucket_name).blob(filename)
            blob.upload_from_string(json.dumps(data), content_type="application/json")

            message.ack()
        except Exception as e:
            print(f"Error processing message: {e}")
            # Implement error handling logic (e.g., retry or log the error)

    streaming_pull_future = subscriber.subscribe(subscription_path, callback=callback)
    print(f"Listening for messages on topic: {topic_name}")

    # Wait for subscription to be cancelled.
    try:
        streaming_pull_future.result()
    except KeyboardInterrupt:
        streaming_pull_future.cancel()
        print("Stopped listening for messages.")


if __name__ == "__main__":
    # Create a Cloud Storage client (assuming authentication is set up)
    storage_client = storage.Client()

    subscribe_to_gumbo_data(storage_client, bucket_name="your-gcs-bucket")