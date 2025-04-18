import os
import time
import logging
from google import genai
from google.genai import types
from google.api_core import exceptions as core_exceptions

# --- Configuration Parameters (Modify these values) ---

# Google Cloud Project Details
PROJECT_ID = "silver-455021"  # Replace with your Project ID
# Attempt to get project ID from environment if not set directly
if not PROJECT_ID or PROJECT_ID == "[your-project-id]":
    PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not PROJECT_ID:
        raise ValueError("Google Cloud Project ID not set. Please set the PROJECT_ID variable.")

LOCATION = "us-central1"          # Veo is typically available in us-central1

# Model Details
VIDEO_MODEL_ID = "veo-2.0-generate-001" # The specific Veo model ID

# Generation Parameters (Customize as needed)
PROMPT = "A cinematic timelapse of a beautiful tropical beach sunrise, waves gently washing ashore."
ASPECT_RATIO = "16:9"     # "16:9" (landscape) or "9:16" (portrait)
DURATION_SECONDS = 8      # Desired video duration (integer 5-8)
NUMBER_OF_VIDEOS = 1      # Number of videos to generate (integer 1-4)
PERSON_GENERATION = "allow_adult" # "allow_adult" or "dont_allow" (use "dont_allow" if you want to avoid generating people)
ENHANCE_PROMPT = True    # Let Gemini enhance the prompt (True/False)
# Optional: Add negative prompt or seed if desired
# NEGATIVE_PROMPT = "low quality, blurry, text"
# SEED = 12345

# Output Video Details
OUTPUT_GCS_BUCKET = "gs://mlb_generated_videos" # Replace with your GCS bucket URI (e.g., "gs://my-video-output-bucket/results")
# The service will create a timestamped sub-directory within this bucket.

# Polling configuration
POLLING_INTERVAL_SECONDS = 60 # How often to check the operation status (in seconds)
POLLING_TIMEOUT_SECONDS = 1800 # Max time to wait for the operation (in seconds, e.g., 30 minutes)

# --- End of Configuration Parameters ---

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_video_from_text(
    project_id: str,
    location: str,
    model_id: str,
    prompt: str,
    output_gcs_uri: str,
    aspect_ratio: str = "16:9",
    duration_seconds: int = 8,
    number_of_videos: int = 1,
    person_generation: str = "allow_adult",
    enhance_prompt: bool = True,
    # negative_prompt: str | None = None, # Uncomment if using
    # seed: int | None = None,           # Uncomment if using
) -> list[str]:
    """
    Generates video(s) from a text prompt using the google-genai SDK for Vertex AI.

    Args:
        project_id: Google Cloud project ID.
        location: Google Cloud region.
        model_id: The Veo model ID.
        prompt: The text prompt describing the video.
        output_gcs_uri: GCS URI prefix for storing the output video(s).
        aspect_ratio: "16:9" or "9:16".
        duration_seconds: Video duration (5-8).
        number_of_videos: How many videos to generate (1-4).
        person_generation: "allow_adult" or "dont_allow".
        enhance_prompt: Whether to enhance the prompt.
        # negative_prompt: Optional negative prompt text.
        # seed: Optional seed for deterministic generation.

    Returns:
        A list of GCS URIs for the generated videos, or an empty list on failure.
    """
    generated_video_uris = []
    try:
        logging.info(f"Initializing GenAI Client for Project: {project_id}, Location: {location}")
        client = genai.Client(vertexai=True, project=project_id, location=location)

        logging.info(f"Using model: {model_id}")

        # --- Configure the generation request ---
        config = types.GenerateVideosConfig(
            aspect_ratio=aspect_ratio,
            output_gcs_uri=output_gcs_uri,
            number_of_videos=number_of_videos,
            duration_seconds=duration_seconds,
            person_generation=person_generation,
            enhance_prompt=enhance_prompt,
            # Uncomment and add if using negative prompt or seed
            # negative_prompt=negative_prompt,
            # seed=seed,
        )

        logging.info(f"Submitting video generation request with prompt: '{prompt}'")
        logging.info(f"Configuration: {config}")

        # --- Start the asynchronous generation process ---
        # This call returns an LRO (Long-Running Operation) object immediately
        operation = client.models.generate_videos(
            model=model_id,
            prompt=prompt,
            config=config,
        )

        logging.info(f"Video generation request submitted. Operation details: {operation.operation}")
        logging.info(f"Polling operation status every {POLLING_INTERVAL_SECONDS} seconds (Timeout: {POLLING_TIMEOUT_SECONDS}s)...")

        start_time = time.time()
        while not operation.done:
            # Check for timeout
            elapsed_time = time.time() - start_time
            if elapsed_time > POLLING_TIMEOUT_SECONDS:
                logging.error(f"Operation timed out after {elapsed_time:.0f} seconds.")
                # You might want to attempt cancellation, though it's best-effort
                # try:
                #    client.operations.cancel(operation)
                #    logging.info("Attempted to cancel the timed-out operation.")
                # except Exception as cancel_err:
                #    logging.warning(f"Could not cancel operation: {cancel_err}")
                return [] # Return empty list on timeout

            # Wait before checking status again
            time.sleep(POLLING_INTERVAL_SECONDS)

            # Refresh the operation status
            # Note: operation.refresh() might exist in some versions,
            # but client.operations.get() is the reliable way shown in the notebook
            try:
                operation = client.operations.get(operation) # Gets the latest status
                logging.info(f"Polling... Current operation state: {operation.operation.state}")
            except core_exceptions.NotFound:
                 logging.error(f"Operation {operation.operation.name} not found during polling. It might have been deleted or the name is incorrect.")
                 return []
            except core_exceptions.ResourceExhausted as quota_ex:
                 logging.warning(f"Quota exceeded while polling operation status: {quota_ex}. Continuing polling after delay...")
                 # Allow the loop to continue, relying on the main timeout
            except Exception as poll_err:
                 logging.error(f"Unexpected error polling operation {operation.operation.name}: {poll_err}")
                 return [] # Exit on unexpected polling error


        # --- Process the completed operation ---
        logging.info("Operation completed.")

        if operation.exception:
            logging.error(f"Operation failed with an exception: {operation.exception}")
            # Check specifically for quota errors in the final status, although they
            # often manifest during the initial request or polling.
            if isinstance(operation.exception, core_exceptions.ResourceExhausted):
                 logging.error("Failure likely due to Quota Exceeded. Please check quotas and consider requesting an increase.")
            elif isinstance(operation.exception, core_exceptions.FailedPrecondition):
                 logging.error("Failure likely due to Failed Precondition. Check if your project is allowlisted for the model/feature.")

        elif operation.response:
            logging.info("Operation successful.")
            # Access results - operation.result contains the response payload
            result_payload = operation.result
            if hasattr(result_payload, 'generated_videos') and result_payload.generated_videos:
                for video_info in result_payload.generated_videos:
                    if hasattr(video_info, 'video') and hasattr(video_info.video, 'uri'):
                        uri = video_info.video.uri
                        logging.info(f"  Generated Video URI: {uri}")
                        generated_video_uris.append(uri)
                    else:
                        logging.warning("  Found video info object, but missing expected URI structure.")
                if not generated_video_uris:
                     logging.warning("Operation successful, but no video URIs found in the result payload.")
            else:
                 logging.warning("Operation successful, but 'generated_videos' attribute missing or empty in the result.")
                 logging.debug(f"Full result payload: {result_payload}")

        else:
            # This case might occur if done is True but response is somehow null/false
            # without an explicit exception being set.
            logging.warning("Operation finished but did not yield a response or an exception.")


    except core_exceptions.ResourceExhausted as e:
        logging.error(f"Quota exceeded during initial request: {e}")
        logging.error("Please check your Vertex AI quotas (requests per minute) for the Veo model.")
        logging.error("Consider adding delays between script runs or requesting a quota increase.")
        # No automatic retry implemented here, unlike the previous script attempt.
        # You could wrap the client.models.generate_videos call in a retry loop if needed.
    except core_exceptions.FailedPrecondition as e:
        logging.error(f"Failed Precondition during initial request: {e}")
        logging.error("This often means the project is not allowlisted for this model/feature.")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        import traceback
        logging.error(traceback.format_exc())

    return generated_video_uris

# --- Main Execution Block ---
if __name__ == "__main__":
    if not OUTPUT_GCS_BUCKET or not OUTPUT_GCS_BUCKET.startswith("gs://"):
         logging.error("Please set a valid GCS bucket URI (starting with gs://) for OUTPUT_GCS_BUCKET.")
    else:
        logging.info("Starting text-to-video generation process using google-genai SDK...")
        script_start_time = time.time()

        generated_uris = generate_video_from_text(
            project_id=PROJECT_ID,
            location=LOCATION,
            model_id=VIDEO_MODEL_ID,
            prompt=PROMPT,
            output_gcs_uri=OUTPUT_GCS_BUCKET,
            aspect_ratio=ASPECT_RATIO,
            duration_seconds=DURATION_SECONDS,
            number_of_videos=NUMBER_OF_VIDEOS,
            person_generation=PERSON_GENERATION,
            enhance_prompt=ENHANCE_PROMPT,
            # Pass negative_prompt=NEGATIVE_PROMPT or seed=SEED here if using
        )

        script_end_time = time.time()
        logging.info(f"Script finished in {script_end_time - script_start_time:.2f} seconds.")

        if generated_uris:
            logging.info("--- Generated Video GCS URIs ---")
            for uri in generated_uris:
                logging.info(f"- {uri}")
            logging.info("---------------------------------")

        else:
            logging.warning("Video generation failed or produced no output URIs. Check logs for details.")