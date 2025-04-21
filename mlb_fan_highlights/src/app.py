import streamlit as st
import json
import logging
from datetime import datetime, UTC, timedelta
import time
from google.cloud import storage # NEW: Import GCS client
from google.api_core import exceptions as google_exceptions
from typing import Dict, List, Tuple, Any, Optional

# --- Import necessary components from your agent script ---
# Assume mlb_agent_graph_refined.py is in the same directory or accessible in PYTHONPATH
try:
    from mlb_agent5 import (
        app,  # The compiled LangGraph app
        TEAMS,
        get_latest_final_game_pk,
        logger, # Use the same logger setup if desired
        load_player_metadata 
        # Add any other specific functions or variables if needed directly
        # Note: We don't need to import *all* functions, only those
        # called directly by the Streamlit app itself.
    )
    # Configure logger for Streamlit if it wasn't configured in the imported script
    if not logger.handlers:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - Streamlit - %(levelname)s - %(message)s')

except ImportError as e:
    st.error(f"Failed to import necessary components from 'mlb_agent_graph_refined.py'. "
             f"Ensure the file exists and is in the correct path. Error: {e}")
    st.stop() # Stop the app if imports fail

# --- Streamlit App Configuration ---
st.set_page_config(page_title="MLB Game Recap Generator", layout="wide")
st.title("⚾ MLB Game Recap Generator")
st.markdown("Select a team to generate a recap of their latest completed game, including script, images, video clips, and audio.")

# --- Logger Setup (Ensure it's configured) ---
# Assuming logger is configured either here or in the imported agent script
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - Streamlit - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__) # Get logger if not imported

# --- NEW: Initialize GCS Client ---
try:
    storage_client = storage.Client()
    logger.info("Initialized Google Cloud Storage client for Signed URLs.")
except Exception as e:
    st.error(f"⚠️ Failed to initialize Google Cloud Storage client: {e}. Media display might fail. Ensure Application Default Credentials are set.")
    storage_client = None # Set to None so checks below fail gracefully
    logger.error(f"Failed to initialize GCS client: {e}", exc_info=True)

# --- Session State Initialization ---
# Used to store results across Streamlit reruns
if 'run_complete' not in st.session_state:
    st.session_state.run_complete = False
if 'final_result' not in st.session_state:
    st.session_state.final_result = None
if 'error_message' not in st.session_state:
    st.session_state.error_message = None
if 'selected_team_key' not in st.session_state:
     st.session_state.selected_team_key = list(TEAMS.keys())[0] # Default selection

# --- Helper Function for Signed URLs ---
def generate_signed_url(gcs_uri: str, expiration_minutes: int = 30) -> Optional[str]:
    """
    Generates a temporary signed URL for a GCS object.

    Args:
        gcs_uri: The gs:// URI of the object (e.g., "gs://bucket/file.jpg").
        expiration_minutes: How long the URL should be valid in minutes.

    Returns:
        The signed HTTPS URL string, or None if an error occurs.
    """
    global storage_client # Access the globally initialized client
    if not storage_client:
        logger.error("GCS storage client not available for generating signed URL.")
        return None
    if not gcs_uri or not gcs_uri.startswith("gs://"):
        logger.warning(f"Invalid GCS URI provided for signing: {gcs_uri}")
        return None

    try:
        # Parse the GCS URI
        bucket_name, blob_name = gcs_uri.replace("gs://", "").split("/", 1)
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        # Check if blob exists (optional, adds an API call but prevents signing non-existent objects)
        # Note: This requires storage.objects.get permission
        # if not blob.exists():
        #     logger.warning(f"Blob does not exist, cannot generate signed URL: {gcs_uri}")
        #     return None

        # Generate the signed URL (v4 signing is recommended)
        # Requires the service account running Streamlit to have
        # 'roles/iam.serviceAccountTokenCreator' on itself OR appropriate permissions.
        # It also needs permission to *read* the object (e.g., roles/storage.objectViewer).
        signed_url = blob.generate_signed_url(
            version="v4",
            expiration=timedelta(minutes=expiration_minutes),
            method="GET",
        )
        logger.debug(f"Generated signed URL for: {gcs_uri}")
        return signed_url

    except google_exceptions.NotFound:
         logger.error(f"Blob not found when trying to generate signed URL: {gcs_uri}")
         return None
    except google_exceptions.Forbidden as e:
         # This is crucial for diagnosing permissions!
         logger.error(f"Permission denied generating signed URL for {gcs_uri}. "
                      f"Ensure the Streamlit service account has 'Service Account Token Creator' role "
                      f"on itself AND 'Storage Object Viewer' role on the bucket/object. Error: {e}", exc_info=False) # Set exc_info=False to avoid huge stack trace in UI logs
         st.warning(f"Permission error generating URL for {gcs_uri}. Check app logs and IAM roles.", icon="🔒")
         return None
    except Exception as e:
        logger.error(f"Error generating signed URL for {gcs_uri}: {e}", exc_info=True)
        return None


# --- Team Selection ---
# Create display names (e.g., "Rangers" from "rangers")
team_display_names = {key: key.replace('_', ' ').title() for key in TEAMS.keys()}
# Reverse lookup for display name -> key
team_key_lookup = {v: k for k, v in team_display_names.items()}

# Get current index based on session state
current_selection_display_name = team_display_names.get(st.session_state.selected_team_key, list(team_display_names.values())[0])
try:
    current_index = list(team_display_names.values()).index(current_selection_display_name)
except ValueError:
    current_index = 0 # Default to first team if something goes wrong


selected_team_display_name = st.selectbox(
    "Select Team:",
    options=list(team_display_names.values()),
    index=current_index,
    key="team_selector", # Use a key to help manage state if needed
    on_change=lambda: st.session_state.update(run_complete=False, final_result=None, error_message=None) # Reset on change
)

# Update the session state key when selection changes
st.session_state.selected_team_key = team_key_lookup[selected_team_display_name]
selected_team_id = TEAMS[st.session_state.selected_team_key]

st.write(f"Selected Team: **{selected_team_display_name}** (ID: {selected_team_id})")

# --- Run Button and Execution Logic ---
if st.button(f"Generate Recap for {selected_team_display_name}'s Latest Game", key="run_button"):
    st.session_state.run_complete = False
    st.session_state.final_result = None
    st.session_state.error_message = None

    status_placeholder = st.empty() # Placeholder for status updates

    with st.spinner(f"Processing request for {selected_team_display_name}... This may take several minutes."):
        try:
            # 1. Find Latest Game PK
            status_placeholder.info("Finding latest completed game...")
            logger.info(f"Attempting to find latest game PK for team ID: {selected_team_id}")
            latest_game_pk = get_latest_final_game_pk(selected_team_id)

            if not latest_game_pk:
                logger.error(f"Could not find a recent completed game for team {selected_team_display_name} (ID: {selected_team_id}).")
                st.session_state.error_message = f"❌ Could not find a recent completed game for {selected_team_display_name}. Please try another team or check game schedules."
                st.session_state.run_complete = True
                status_placeholder.empty() # Clear status message
                st.rerun() # Rerun to display error message immediately
            # --- *** ADD PLAYER METADATA LOADING HERE *** ---
            status_placeholder.info("Loading player metadata...")
            try:
                 player_lookup = load_player_metadata() # Call the function
                 if not player_lookup:
                     logger.warning("Player metadata lookup returned empty from agent function.")
                     # Decide how critical this is - maybe show a warning?
                     st.warning("⚠️ Could not load player metadata. Headshot images may be unavailable.")
                 else:
                     logger.info(f"Successfully loaded {len(player_lookup)} players via Streamlit.")
            except Exception as load_err:
                 logger.error(f"Error calling load_player_metadata from Streamlit: {load_err}", exc_info=True)
                 st.error("Failed to load player metadata. Headshots may be unavailable.")
                 player_lookup = {} # Ensure it's an empty dict on error
            # --- *** END PLAYER METADATA LOADING *** ---

            logger.info(f"Found latest game PK: {latest_game_pk} for team {selected_team_display_name}")
            status_placeholder.info(f"Found latest game (PK: {latest_game_pk}). Preparing agent...")

            # 2. Prepare Initial State
            # Keep this relatively simple; the agent loads most data itself.
            # We primarily need to pass the trigger (game_pk) and task.
            task = f"Provide a detailed recap of game {latest_game_pk} ({selected_team_display_name}), highlighting impactful plays and player performances in an engaging two-host dialogue format."
            initial_state = {
                "task": task,
                "game_pk": latest_game_pk,
                "max_revisions": 2, # Default from your script
                "revision_number": 0,
                "player_lookup_dict": player_lookup, # Agent loads this if needed
                "narrative_context": [],
                "all_image_assets": [],
                "all_video_assets": [],
                "generated_visual_assets": [],
                "generated_video_assets": [],
                "visual_revision_number": 0,
                "max_visual_revisions": 2, # Default from your script
                "error": None,
                # Add other required keys with default/None values if compilation fails without them
                "plan": None,
                "structured_data": None,
                "image_search_queries":None,
                "retrieved_image_data":None,
                "draft": None,
                "critique": None,
                "generated_content": None,
                "visual_generation_prompts": [],
                "visual_critique": None,
                "generated_audio_uri": None, # Important: Initialize audio URI
            }
            logger.info(f"Initial state prepared for game PK {latest_game_pk}.")

            # 3. Invoke Agent Graph
            status_placeholder.info(f"Running MLB Agent for game {latest_game_pk}... (This can take minutes due to visual/audio generation)")
            start_time = time.time()

            # Set a reasonable recursion limit based on your graph structure
            recursion_limit = 50 # Adjust as needed

            final_state = app.invoke(initial_state, {"recursion_limit": recursion_limit})
            end_time = time.time()
            logger.info(f"Agent invocation finished in {end_time - start_time:.2f} seconds.")

            # 4. Store Results or Errors
            if final_state and isinstance(final_state, dict):
                 st.session_state.final_result = final_state
                 if final_state.get("error"):
                     st.session_state.error_message = f"Agent completed with an error: {final_state['error']}"
                     logger.error(f"Agent finished with error for game {latest_game_pk}: {final_state['error']}")
                 else:
                      logger.info(f"Agent completed successfully for game {latest_game_pk}.")
                      status_placeholder.success(f"Recap generated successfully for game {latest_game_pk}!")
            else:
                st.session_state.error_message = "❌ Agent did not return a valid result dictionary."
                logger.error(f"Agent returned invalid result type for game {latest_game_pk}: {type(final_state)}")

        except Exception as e:
            logger.error(f"An unexpected error occurred during agent execution for team {selected_team_display_name}: {e}", exc_info=True)
            st.session_state.error_message = f"❌ An unexpected error occurred: {e}"

        finally:
             st.session_state.run_complete = True
             status_placeholder.empty() # Clear status message
             st.rerun() # Rerun to display results or final error

# --- Display Results ---
if st.session_state.run_complete:
    if st.session_state.error_message:
        st.error(st.session_state.error_message)
    elif st.session_state.final_result:
        results = st.session_state.final_result
        st.success(f"Recap Generation Complete for Game PK: {results.get('game_pk', 'N/A')}")

        # --- Display Generated Script ---
        st.subheader("🎙️ Generated Dialogue Script")
        script_content = results.get("generated_content")
        if script_content:
            # Split lines and add speaker hints for readability in Streamlit
            script_lines = script_content.strip().split('\n')
            display_script = ""
            for i, line in enumerate(script_lines):
                 speaker = "Host 1:" if i % 2 == 0 else "Host 2:"
                 display_script += f"**{speaker}** {line}\n\n" # Add markdown bold and spacing
            st.markdown(display_script)
            # st.text_area("Script", script_content, height=300) # Alternative display
        else:
            st.warning("No script content was generated.")


        # --- Display Generated Audio ---
        st.subheader("🎧 Generated Audio")
        audio_uri = results.get("generated_audio_uri")
        if audio_uri:
            st.write(f"Audio GCS URI: `{audio_uri}`") # Still show the original URI
            if isinstance(audio_uri, str) and audio_uri.startswith("gs://"):
                 signed_audio_url = generate_signed_url(audio_uri) # Generate Signed URL
                 if signed_audio_url:
                     try:
                         st.audio(signed_audio_url) # Use the signed URL
                         st.caption(f"Audio player loaded via temporary URL (expires).")
                     except Exception as audio_err:
                         st.warning(f"Could not display audio player for {audio_uri}. Error: {audio_err}")
                         st.markdown(f"Direct Link (expires): [Listen to Audio]({signed_audio_url})") # Fallback link
                 else:
                     st.warning(f"Could not generate temporary URL to play audio from: {audio_uri}")
            else:
                st.warning(f"Invalid or non-GCS audio URI format: {audio_uri}")
        else:
            st.warning("No audio URI found in the results.")

        # --- Display Visual Assets (Images) ---
        st.subheader("🖼️ Visual Assets (Images)")
        all_image_assets = results.get("all_image_assets")
        if all_image_assets and isinstance(all_image_assets, list):
            if not all_image_assets:
                 st.info("No image assets were generated or retrieved.")
            else:
                 # Create columns for better layout
                 cols = st.columns(3) # Adjust number of columns as needed
                 col_idx = 0
                 for i, asset in enumerate(all_image_assets):
                    with cols[col_idx % len(cols)]:
                         st.markdown(f"**Image {i+1}**")
                         asset_type = asset.get("type", "N/A").replace("_", " ").title()
                         model = asset.get("model_used", "")
                         source_info = ""
                         # ... (your existing logic to determine source_info) ...
                         if asset_type == "Generated Image":
                             source_info = f"Generated ({model}) from prompt: *'{asset.get('prompt_origin', 'N/A')[:50]}...'*"
                         elif asset_type == "Headshot":
                             source_info = f"Static Headshot for: *{asset.get('entity_name', 'N/A')}*"
                         elif asset_type == "Logo":
                              source_info = f"Static Logo Search: *'{asset.get('search_term_origin', 'N/A')}'*"
                         else:
                             # Handle cases where 'type' might be missing or different
                             source_info = f"Type: {asset.get('image_type', asset_type)}" # Fallback to image_type if 'type' missing

                         st.markdown(f"_{source_info}_")

                         img_uri = asset.get("image_uri")
                         st.write(f"GCS URI: `{img_uri}`") # Always show original URI

                         if img_uri and isinstance(img_uri, str) and img_uri.startswith("gs://"):
                              signed_image_url = generate_signed_url(img_uri) # Generate Signed URL
                              if signed_image_url:
                                  try:
                                      st.image(signed_image_url, width=200) # Use Signed URL
                                      st.caption(f"Image loaded via temporary URL (expires).")
                                  except Exception as img_err:
                                      st.warning(f"Could not display image for {img_uri}. Error: {img_err}")
                                      st.markdown(f"Direct Link (expires): [View Image]({signed_image_url})") # Fallback link
                              else:
                                   st.warning(f"Could not generate temporary URL to display image from: {img_uri}")
                         elif img_uri: # Handle cases where it might be a direct HTTPS URL already?
                              st.warning(f"Image URI is not a GCS URI, attempting direct display: {img_uri}")
                              try:
                                  st.image(img_uri, width=200)
                              except Exception as img_err:
                                  st.error(f"Failed to display non-GCS image URI. Error: {img_err}")
                         else:
                              st.warning("Image URI missing in asset data.")
                         st.divider() # Separator between images in a column
                    col_idx += 1
        else:
            st.warning("Image assets data is missing or not a list in the results.")

        # --- Display Visual Assets (Videos) ---
        st.subheader("🎬 Visual Assets (Videos)")
        all_video_assets = results.get("all_video_assets")
        if all_video_assets and isinstance(all_video_assets, list):
            if not all_video_assets:
                st.info("No video assets were generated.")
            else:
                cols_vid = st.columns(2) # Adjust number of columns
                col_idx_vid = 0
                for i, asset in enumerate(all_video_assets):
                     with cols_vid[col_idx_vid % len(cols_vid)]:
                         st.markdown(f"**Video Clip {i+1}**")
                         model = asset.get("model_used", "N/A")
                         prompt = asset.get("source_prompt", 'N/A')
                         st.markdown(f"Generated ({model}) from prompt: *'{prompt[:60]}...'*")

                         vid_uri = asset.get("video_uri")
                         st.write(f"GCS URI: `{vid_uri}`") # Always show original URI

                         if vid_uri and isinstance(vid_uri, str) and vid_uri.startswith("gs://"):
                             signed_video_url = generate_signed_url(vid_uri) # Generate Signed URL
                             if signed_video_url:
                                 try:
                                     st.video(signed_video_url) # Use Signed URL
                                     st.caption(f"Video loaded via temporary URL (expires).")
                                 except Exception as vid_err:
                                     st.warning(f"Could not display video player for {vid_uri}. Error: {vid_err}")
                                     st.markdown(f"Direct Link (expires): [View Video]({signed_video_url})") # Fallback link
                             else:
                                 st.warning(f"Could not generate temporary URL to display video from: {vid_uri}")
                         elif vid_uri:
                              st.warning(f"Video URI is not a GCS URI, attempting direct display: {vid_uri}")
                              try:
                                  st.video(vid_uri)
                              except Exception as vid_err:
                                  st.error(f"Failed to display non-GCS video URI. Error: {vid_err}")
                         else:
                              st.warning("Video URI missing in asset data.")

                         st.divider()
                     col_idx_vid += 1
        else:
            st.warning("Video assets data is missing or not a list in the results.")


        # --- Optionally display raw final state for debugging ---
        with st.expander("Show Raw Final State (for Debugging)"):
            # Exclude potentially large fields for display
            debug_state = {k: v for k, v in results.items() if k not in [
                'structured_data', 'narrative_context', 'draft',
                'generated_content', 'all_image_assets', 'all_video_assets',
                'generated_visual_assets', 'generated_video_assets',
                'player_lookup_dict']}
            st.json(json.dumps(debug_state, indent=2, default=str))

elif not st.session_state.run_complete and not st.session_state.error_message:
    st.info("Select a team and click the button above to generate the game recap.")

# --- Footer or additional info ---
st.markdown("---")
st.markdown("Powered by LangGraph, Vertex AI (Gemini, Imagen, Veo, TTS), BigQuery, and GCS.")