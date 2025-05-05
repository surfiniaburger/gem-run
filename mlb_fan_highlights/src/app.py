# mlb_agent_ui_updated.py
import streamlit as st
import json
import logging
from datetime import datetime, timedelta, UTC
import time
from typing import Optional, Dict, Any, List
import os 
# --- GCS Client & Credentials ---
from google.cloud import storage
from google.api_core import exceptions as google_exceptions
from google.cloud import secretmanager
from google.oauth2 import service_account

# --- Import necessary components from your NEW agent script ---
# *** Make sure this points to your latest agent script file ***
try:
    from mlb_agentz import (
        app,  # The compiled LangGraph app
        TEAMS,
        get_latest_final_game_pk,
        load_player_metadata,
        # Define load_player_metadata if it exists in your agent script,
        # otherwise load it here or remove if not needed for initial state
        # load_player_metadata, # Example import if defined in agent
        logger, GCP_PROJECT_ID # Use the same logger setup if desired
    )
    # Configure logger for Streamlit if it wasn't configured in the imported script
    if not logger.handlers:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - Streamlit - %(levelname)s - %(message)s')
        logger = logging.getLogger(__name__) # Define logger if not imported

except ImportError as e:
    st.error(f"Failed to import necessary components from 'mlb_agent5.py'. "
             f"Ensure the file exists and is in the correct path. Error: {e}")
    st.stop() # Stop the app if imports fail
except NameError as ne:
    # Handle if logger wasn't defined/imported
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - Streamlit - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.warning(f"Caught NameError during import (likely 'logger'): {ne}. Initialized logger.")

# --- Configuration ---
SERVICE_ACCOUNT_SECRET_ID = "streamlit-gcs-sa-key" # <-- **REPLACE** with your Secret ID for the SA key JSON
SECRET_VERSION = "latest"
# Ensure GCP_PROJECT_ID is available (imported above)
if 'GCP_PROJECT_ID' not in globals():
     st.error("GCP_PROJECT_ID is not defined. Please define it or import from agent.")
     st.stop()

# --- Load Credentials for Signed URLs ---
@st.cache_resource # Cache credentials for the session duration
def load_gcs_signing_credentials(project_id: str, secret_id: str, secret_version: str) -> Optional[service_account.Credentials]:
    """Loads Service Account credentials from Secret Manager for signing URLs."""
    try:
        client = secretmanager.SecretManagerServiceClient()
        name = f"projects/{project_id}/secrets/{secret_id}/versions/{secret_version}"
        response = client.access_secret_version(request={"name": name})
        sa_key_json = response.payload.data.decode("UTF-8")
        sa_key_dict = json.loads(sa_key_json)

        # Scope needed for signing URLs and potentially reading objects if client uses it
        credentials = service_account.Credentials.from_service_account_info(
            sa_key_dict,
            scopes=['https://www.googleapis.com/auth/devstorage.read_only']
        )
        logger.info(f"Successfully loaded signing credentials from secret: {secret_id}")
        return credentials
    except Exception as e:
        logger.error(f"Failed to load signing credentials from Secret Manager ({secret_id}): {e}", exc_info=True)
        st.error(f"⚠️ Failed to load credentials from Secret Manager ('{secret_id}') required for media display. Ensure Secret exists and the app's Service Account has the 'Secret Manager Secret Accessor' role on this secret.")
        return None

# --- Initialize GCS Client & Load Credentials ---
gcs_signing_credentials = load_gcs_signing_credentials(GCP_PROJECT_ID, SERVICE_ACCOUNT_SECRET_ID, SECRET_VERSION)

try:
    # Initialize a storage client - can use ADC or the loaded credentials
    # Using ADC is often simpler if the Cloud Run/App Engine SA has Storage permissions
    storage_client = storage.Client(project=GCP_PROJECT_ID)
    # Or use the key: storage_client = storage.Client(project=GCP_PROJECT_ID, credentials=gcs_signing_credentials)
    logger.info("Initialized Google Cloud Storage client.")
except Exception as e:
    st.error(f"⚠️ Failed to initialize Google Cloud Storage client: {e}. Media display might fail.")
    storage_client = None
    logger.error(f"Failed to initialize GCS client: {e}", exc_info=True)


# --- Streamlit App Configuration ---
# Use wide layout for better display
st.title("⚾ MLB Game Recap Video Generator")
st.markdown("Select a team to generate a video recap of their latest completed game.")

# --- Session State Initialization ---
if 'run_complete' not in st.session_state:
    st.session_state.run_complete = False
if 'final_state' not in st.session_state: # Renamed from final_result
    st.session_state.final_state = None
if 'error_message' not in st.session_state:
    st.session_state.error_message = None
if 'selected_team_key' not in st.session_state:
     st.session_state.selected_team_key = list(TEAMS.keys())[0] # Default selection

# --- Helper Function for Signed URLs ---
def generate_signed_url(gcs_uri: str, expiration_minutes: int = 30) -> Optional[str]:
    """Generates a temporary signed URL for a GCS object using loaded credentials."""
    global storage_client, gcs_signing_credentials
    if not storage_client:
        logger.error("GCS storage client not available for generating signed URL.")
        return None
    if not gcs_signing_credentials: # Check if credentials loaded successfully
        logger.error("Signing credentials not available for generating signed URL.")
        # Error message already shown by load_gcs_signing_credentials
        return None
    if not gcs_uri or not gcs_uri.startswith("gs://"):
        logger.warning(f"Invalid GCS URI provided for signing: {gcs_uri}")
        return None

    try:
        bucket_name, blob_name = gcs_uri.replace("gs://", "").split("/", 1)
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        # Check if blob exists before generating URL
        if not blob.exists():
             logger.error(f"Blob not found when trying to generate signed URL: {gcs_uri}")
             return None

        signed_url = blob.generate_signed_url(
            version="v4",
            expiration=timedelta(minutes=expiration_minutes),
            method="GET",
            credentials=gcs_signing_credentials, # Explicitly use the loaded credentials
        )
        logger.debug(f"Generated signed URL for: {gcs_uri}")
        return signed_url
    # Catch specific exceptions for better feedback
    except google_exceptions.NotFound: # Already checked with blob.exists(), but keep for safety
         logger.error(f"Blob not found (double check) generating signed URL: {gcs_uri}")
         return None
    except google_exceptions.Forbidden as e:
         # This error often means the SA used for signing lacks 'roles/iam.serviceAccountTokenCreator'
         # OR the GCS client's SA lacks 'roles/storage.objectViewer' on the bucket.
         logger.error(f"Permission denied generating signed URL for {gcs_uri}. Check Service Account Token Creator role ON SA '{gcs_signing_credentials.service_account_email}' and Storage Object Viewer role ON bucket '{bucket_name}'. Error: {e}", exc_info=False)
         st.warning(f"Permission error generating URL for `{os.path.basename(gcs_uri)}`. Check app logs and required IAM roles.", icon="🔒")
         return None
    except Exception as e:
        logger.error(f"Error generating signed URL for {gcs_uri}: {e}", exc_info=True)
        return None


# --- Agent Streaming Function ---
def run_agent_and_stream_progress(agent_app, initial_agent_state):
    """
    Runs the agent using app.stream() and yields formatted progress strings.
    Stores the FINAL complete state in st.session_state.
    """
    latest_full_state = None # Variable to hold the most recent complete state dictionary
    final_state_on_success = None
    nodes_seen_in_stream = []
    # Clear previous final state and error message managed by the stream function
    st.session_state['final_state'] = None
    st.session_state['error_message'] = None # Stream function will set this on error

    # Updated node emojis including new steps
    node_emojis = {
        "planner": "📅", "retrieve_data": "🔍", "generate": "✍️",
        "reflect": "🤔", "research_critique": "📚", "web_search_context": "🌐",
        "analyze_script_for_images": "🖼️", "retrieve_images": "🏞️",
        "analyze_script_for_visual_prompts": "🎬", "generate_visuals": "✨",
        "critique_visuals": "🧐", "prepare_new_visual_prompts": "📝",
        "generate_video_clips": "🎞️", # Using film reel
        "generate_audio": "🔊",
        "transcribe_for_timestamps": "⏱️", # NEW: Stopwatch
        "aggregate_final_output": "📦",
        "assemble_video": "🛠️", # NEW: Hammer and Wrench for assembly
        "__end__": "🏁"
    }

    try:
        recursion_limit = 50 # Increased limit for the deeper graph
        logger.info(f"Starting agent stream with recursion limit: {recursion_limit}...")
        # Use app.stream which yields updates for each node completion
        stream = agent_app.stream(initial_agent_state, {"recursion_limit": recursion_limit}, stream_mode="updates")

        yield "🚀 **Agent execution started...** (This may take several minutes)\n\n---\n" # Initial message

        for step_update in stream:
            # step_update is a dictionary like {'node_name': full_state_after_node}
            if not step_update or not isinstance(step_update, dict): # Handle potential empty updates
                logger.debug("Stream yielded empty or non-dict update, skipping.")
                continue

            # --- Safely extract node name and state ---
            try:
                 node_name = list(step_update.keys())[0]
                 latest_full_state = step_update[node_name] # Capture the full state dictionary
                 if not isinstance(latest_full_state, dict):
                      logger.warning(f"Received non-dict state for node '{node_name}', using previous state if available.")
                      # Keep the previous state if the current one is invalid
                      latest_full_state = latest_full_state if latest_full_state else {}
                      # Decide how to proceed - maybe yield a warning? For now, log and continue.
            except (IndexError, TypeError) as ex:
                 logger.error(f"Error parsing stream update: {step_update}. Error: {ex}")
                 continue # Skip this update if structure is wrong
            # --- End Safe Extraction ---

            nodes_seen_in_stream.append(node_name) # Log node name
            logger.info(f"STREAM UI: Received update from node: {node_name}")

            emoji = node_emojis.get(node_name, "⚙️") # Default emoji
            node_error = latest_full_state.get("error") # Check for error IN the state

            status_message = f"{emoji} **{node_name.replace('_', ' ').title()}**"
            if node_error:
                 status_message += f" reported error: `{str(node_error)[:100]}...`"
                 logger.warning(f"Node '{node_name}' reported error in stream state: {node_error}")
                 # Set the main error message if a node reports one, allowing loop to continue
                 # but flagging the issue
                 st.session_state['error_message'] = f"Error in node '{node_name}': {node_error}"
                 # Don't immediately break; the graph might handle the error or proceed
            else:
                 status_message += " finished."

            # Optional: Add specific details for key nodes based on state
            if node_name == "generate":
                status_message += f" (Script Revision {latest_full_state.get('revision_number', 0)})" # Rev # starts at 0
            elif node_name == "generate_visuals":
                vis_rev = latest_full_state.get('visual_revision_number', 0) # generate_visuals increments *before* returning
                status_message += f" (Visuals Revision {max(0, vis_rev - 1)} generated)" # Show rev that just finished
            elif node_name == "retrieve_data":
                 narrative_ctx = latest_full_state.get('narrative_context')
                 ctx_len = len(narrative_ctx) if isinstance(narrative_ctx, list) else 0
                 struct_data = latest_full_state.get('structured_data')
                 struct_len = len(struct_data) if isinstance(struct_data, list) else 0
                 status_message += f" (Context: {ctx_len}, Structured: {struct_len})"
            elif node_name == "retrieve_images":
                 retrieved_images = latest_full_state.get('retrieved_image_data')
                 img_len = len(retrieved_images) if isinstance(retrieved_images, list) else 0
                 status_message += f" (Found {img_len} static assets)"
            elif node_name == "generate_visuals":
                 gen_visuals = latest_full_state.get('generated_visual_assets')
                 vis_len = len(gen_visuals) if isinstance(gen_visuals, list) else 0
                 status_message += f" (Now {vis_len} total generated images)"
            elif node_name == "generate_video_clips":
                 gen_videos = latest_full_state.get('generated_video_assets')
                 vid_len = len(gen_videos) if isinstance(gen_videos, list) else 0
                 status_message += f" (Now {vid_len} total generated videos)"
            elif node_name == "generate_audio":
                if latest_full_state.get("generated_audio_uri"):
                    status_message += " (Audio file ready)"
            elif node_name == "transcribe_for_timestamps":
                 timestamps = latest_full_state.get("word_timestamps")
                 ts_len = len(timestamps) if isinstance(timestamps, list) else 0
                 if ts_len > 0:
                    status_message += f" (Found {ts_len} word timestamps)"
                 else:
                     status_message += " (Timestamp generation issue?)"
            elif node_name == "assemble_video":
                 if latest_full_state.get("final_video_uri"):
                     status_message += " (Final video created!)"
                 else:
                     status_message += " (Video assembly in progress...)"

            yield status_message + "\n\n---\n" # Add divider for clarity
            # time.sleep(0.1) # Optional small delay

        # ---- After the loop completes WITHOUT raising an exception ----
        final_state_on_success = latest_full_state # The state after the last node finished
        st.session_state['final_state'] = final_state_on_success # Store the complete final state
        logger.info(f"STREAM UI: Stream finished successfully. Nodes seen: {nodes_seen_in_stream}")
        if st.session_state['final_state']:
            logger.info(f"STREAM UI: Storing final state with keys: {list(st.session_state['final_state'].keys())}")
            yield f"\n{node_emojis['__end__']} **Agent execution complete! Processing final video...**"
        else:
             logger.error("STREAM UI: Stream finished but final state is missing!")
             st.session_state['error_message'] = "Execution finished, but final state is missing. Check logs."
             yield f"\n⚠️ **Execution finished, but final state is missing. Check logs.**"


    except Exception as e:
        # ---- If an exception occurs DURING the stream execution (e.g., network error, internal LangGraph error) ----
        logger.error(f"STREAM UI: Critical error during agent stream: {e}", exc_info=True)
        logger.warning(f"STREAM UI: Stream errored. Nodes seen before error: {nodes_seen_in_stream}")
        # Store the partial state captured just before the error
        st.session_state['final_state'] = latest_full_state
        # Store the critical error message
        st.session_state['error_message'] = f"Critical Error during agent execution: {e}"
        logger.warning(f"STREAM UI: Storing partial state on error with keys: {list(st.session_state['final_state'].keys()) if st.session_state['final_state'] else 'None'}")
        yield f"\n❌ **{st.session_state['error_message']}**"

    # This block always runs after the try/except completes (success or failure)
    finally:
        st.session_state.run_complete = True
        # No need to rerun here, the display logic runs based on run_complete=True


# --- UI Layout ---
col1, col2 = st.columns([1, 3]) # Left column for controls, right for output

with col1:
    st.subheader("Controls")
    # --- Team Selection ---
    team_display_names = {key: key.replace('_', ' ').title() for key in TEAMS.keys()}
    team_key_lookup = {v: k for k, v in team_display_names.items()}
    current_selection_display_name = team_display_names.get(st.session_state.selected_team_key, list(team_display_names.values())[0])
    try:
        current_index = list(team_display_names.values()).index(current_selection_display_name)
    except ValueError:
        current_index = 0

    selected_team_display_name = st.selectbox(
        "Select Team:",
        options=list(team_display_names.values()),
        index=current_index,
        key="team_selector",
        # Reset state only when the selection *changes* and the button is pressed again
        # The button press itself handles the main reset now.
        # on_change=lambda: st.session_state.update(run_complete=False, final_state=None, error_message=None)
    )

    st.session_state.selected_team_key = team_key_lookup[selected_team_display_name]
    selected_team_id = TEAMS[st.session_state.selected_team_key]
    st.write(f"Team ID: `{selected_team_id}`")

    # --- Run Button ---
    if st.button(f"🚀 Generate Video for {selected_team_display_name}", key="run_button", use_container_width=True):
        # Reset state variables for a new run
        st.session_state.run_complete = False
        st.session_state.final_state = None
        st.session_state.error_message = None
        # Intentionally NOT clearing selected_team_key here

        # Check credentials BEFORE starting the long process
        if not gcs_signing_credentials:
             st.session_state.error_message = "Failed to load necessary credentials from Secret Manager. Cannot generate signed URLs for media display."
             st.session_state.run_complete = True
             st.rerun() # Rerun to show the error immediately

        # Placeholder for the stream output in the right column
        with col2:
             stream_output_container = st.container(border=True, height=400)
             progress_bar = st.progress(0, text="Agent starting...") # Optional progress bar

        # --- Start Agent Execution ---
        try:
            # 1. Find Latest Game PK
            with stream_output_container: st.info("Finding latest completed game...")
            logger.info(f"Attempting to find latest game PK for team ID: {selected_team_id}")
            latest_game_pk = get_latest_final_game_pk(selected_team_id)

            if not latest_game_pk:
                st.session_state.error_message = f"❌ Could not find a recent completed game for {selected_team_display_name}."
                st.session_state.run_complete = True
                st.rerun() # Stop and show error

            # 2. Load Player Metadata
            with stream_output_container: st.info("Loading player metadata...")
            try:
                player_lookup = load_player_metadata() # Use the imported function
                if not player_lookup:
                     with stream_output_container: st.warning("⚠️ Could not load player metadata (will proceed without it).")
                else:
                    logger.info(f"Successfully loaded {len(player_lookup)} players for lookup.")
            except Exception as load_err:
                logger.error(f"Error loading player metadata: {load_err}", exc_info=True)
                with stream_output_container: st.warning("⚠️ Failed to load player metadata (will proceed without it).")
                player_lookup = {}

            with stream_output_container: st.info(f"Found latest game (PK: {latest_game_pk}). Preparing agent...")

            # 3. Prepare Initial State - Match the AgentState definition
            task = f"Provide a detailed video recap of game {latest_game_pk} ({selected_team_display_name}), highlighting impactful plays and player performances in an engaging two-host dialogue format, synchronized with visuals."
            initial_state = {
                "task": task,
                "game_pk": latest_game_pk,
                "max_revisions": 2,         # Max text revisions
                "revision_number": 0,
                "player_lookup_dict": player_lookup,
                "narrative_context": [],
                "all_image_assets": [],     # Final combined images
                "all_video_assets": [],     # Final combined video CLIPS (intermediate)
                "generated_visual_assets": [], # Imagen generations
                "generated_video_assets": [], # Veo generations (intermediate)
                "visual_revision_number": 0,
                "max_visual_revisions": 2,  # Max visual generation loops
                "error": None,
                "plan": None,
                "structured_data": None,
                "image_search_queries": None,
                "retrieved_image_data": None, # Static assets
                "draft": None,                # Current script draft
                "critique": None,
                "generated_content": None,    # Final script
                "visual_generation_prompts": [],
                "visual_critique": None,
                "generated_audio_uri": None,  # URI for TTS audio
                "word_timestamps": None,      # List from STT
                "final_video_uri": None       # URI for final assembled video
            }
            logger.info(f"Initial state prepared for game PK {latest_game_pk}.")
            logger.debug(f"Initial state keys: {list(initial_state.keys())}")

            # 4. Execute Agent via Streaming
            start_time = time.time()
            with col2.container(): # Use the main right column for final results display later
                 with stream_output_container: # Stream progress into the dedicated box
                     # Update progress bar within the stream loop if possible, or just show activity
                     st.write_stream(run_agent_and_stream_progress(app, initial_state))
                 # Once stream is done, indicate completion time
                 end_time = time.time()
                 logger.info(f"Agent streaming process finished UI-side in {end_time - start_time:.2f} seconds.")
                 # Progress bar complete
                 if 'progress_bar' in locals(): progress_bar.progress(1.0, text="Agent finished!")

        except Exception as e:
            # Catch errors happening *before* the stream even starts
            logger.error(f"An unexpected error occurred *before* agent streaming could start: {e}", exc_info=True)
            st.session_state.error_message = f"❌ An unexpected error occurred before agent start: {e}"
            st.session_state.run_complete = True # Mark as complete to show error
            st.rerun() # Rerun to display the error message immediately

        # --- IMPORTANT ---
        # After the button logic finishes (including the stream), Streamlit naturally
        # reruns the script from the top. The display logic below will then execute
        # because st.session_state.run_complete is now True.
        # We added st.rerun() only in specific error cases above to force immediate display.


with col2:
    st.subheader("Results")
    results_container = st.container()

    # --- Display Results ---
    if st.session_state.run_complete:
        with results_container:
            # Check for errors FIRST (set by stream function or outer try/except)
            if st.session_state.get("error_message"):
                st.error(st.session_state.error_message)
                # Optionally stop here if error means no results are useful
                # st.stop()

            # Check if we have a final state dictionary to process
            final_agent_state = st.session_state.get('final_state')

            if final_agent_state and isinstance(final_agent_state, dict):
                st.success(f"Recap Generation Process Complete for Game PK: {final_agent_state.get('game_pk', 'N/A')}")

                # --- PRIORITY: Display Final Assembled Video ---
                st.subheader("🏁 Final Assembled Video")
                final_video_uri = final_agent_state.get("final_video_uri")
                if final_video_uri:
                    st.write(f"Video GCS URI: `{final_video_uri}`")
                    if isinstance(final_video_uri, str) and final_video_uri.startswith("gs://"):
                         signed_final_video_url = generate_signed_url(final_video_uri, expiration_minutes=60) # Longer expiry for video
                         if signed_final_video_url:
                             try:
                                 st.video(signed_final_video_url)
                                 st.caption(f"Final video loaded via temporary URL (expires).")
                             except Exception as final_vid_err:
                                 st.warning(f"Could not display final video player ({final_video_uri}). Error: {final_vid_err}")
                                 st.markdown(f"Direct Link (expires): [Watch Final Video]({signed_final_video_url})")
                         else:
                             st.error(f"Could not generate temporary URL to display the final video from: `{final_video_uri}`. Please check GCS permissions and app logs.")
                    else:
                        st.warning(f"Final video URI has an invalid or non-GCS format: {final_video_uri}")
                else:
                    st.warning("No final assembled video URI found in the final state. The process might have failed during assembly.")
                    # Display error from state if it exists and final_video_uri is missing
                    assembly_error = final_agent_state.get("error")
                    if assembly_error and not final_video_uri :
                         st.error(f"Assembly Error reported: {assembly_error}")


                # --- Display Intermediate Assets (Optional / Debug) ---
                with st.expander("Show Intermediate Assets (Script, Audio, Images, Clips)"):

                    # --- Display Generated Script ---
                    st.subheader("🎙️ Generated Dialogue Script")
                    script_content = final_agent_state.get("generated_content")
                    if script_content:
                        script_lines = script_content.strip().split('\n')
                        display_script = ""
                        for i, line in enumerate(script_lines):
                             if line.strip():
                                 speaker = "Host 1:" if i % 2 == 0 else "Host 2:"
                                 display_script += f"**{speaker}** {line}\n\n"
                        st.markdown(display_script)
                    else:
                        st.info("No final script content was generated or found.")

                    # --- Display Generated Audio ---
                    st.subheader("🎧 Generated Audio")
                    audio_uri = final_agent_state.get("generated_audio_uri")
                    if audio_uri:
                        st.write(f"Audio GCS URI: `{audio_uri}`")
                        if isinstance(audio_uri, str) and audio_uri.startswith("gs://"):
                             signed_audio_url = generate_signed_url(audio_uri)
                             if signed_audio_url:
                                 try:
                                     st.audio(signed_audio_url)
                                     st.caption(f"Audio player loaded via temporary URL (expires).")
                                 except Exception as audio_err:
                                     st.warning(f"Could not display audio player for {audio_uri}. Error: {audio_err}")
                                     st.markdown(f"Direct Link (expires): [Listen to Audio]({signed_audio_url})")
                             else:
                                 st.warning(f"Could not generate temporary URL to play audio from: {audio_uri}")
                        else:
                            st.warning(f"Invalid or non-GCS audio URI format: {audio_uri}")
                    else:
                        st.info("No audio URI found in the final state.")

                    # --- Display Visual Assets (Images) ---
                    st.subheader("🖼️ Intermediate Visual Assets (Images)")
                    all_image_assets = final_agent_state.get("all_image_assets") # Combined static + generated
                    if all_image_assets and isinstance(all_image_assets, list):
                        if not all_image_assets:
                             st.info("No image assets were found or generated.")
                        else:
                             cols = st.columns(4) # Use more columns potentially
                             col_idx = 0
                             for i, asset in enumerate(all_image_assets):
                                with cols[col_idx % len(cols)]:
                                     # Simplified display for expander
                                     asset_type = asset.get("type", asset.get("image_type", "N/A")).replace("_", " ").title()
                                     img_uri = asset.get("image_uri")
                                     st.markdown(f"**{asset_type} {i+1}**")
                                     if img_uri and isinstance(img_uri, str) and img_uri.startswith("gs://"):
                                          signed_image_url = generate_signed_url(img_uri, expiration_minutes=10) # Shorter expiry ok
                                          if signed_image_url:
                                              st.image(signed_image_url, width=150, caption=f"{os.path.basename(img_uri)}")
                                          else:
                                               st.warning(f"No URL for {os.path.basename(img_uri)}")
                                     else:
                                          st.warning("No valid GCS URI.")
                                     # st.divider() # Less clutter in expander
                                col_idx += 1
                    else:
                        st.info("Image assets data is missing or not a list in the final state.")

                    # --- Display Visual Assets (Videos) ---
                    st.subheader("🎬 Intermediate Visual Assets (Video Clips)")
                    all_video_clips = final_agent_state.get("all_video_assets") # Intermediate clips
                    if all_video_clips and isinstance(all_video_clips, list):
                        if not all_video_clips:
                            st.info("No intermediate video clips were generated.")
                        else:
                            cols_vid = st.columns(3) # Can fit more clips potentially
                            col_idx_vid = 0
                            for i, asset in enumerate(all_video_clips):
                                 with cols_vid[col_idx_vid % len(cols_vid)]:
                                     st.markdown(f"**Clip {i+1}**")
                                     vid_uri = asset.get("video_uri")
                                     if vid_uri and isinstance(vid_uri, str) and vid_uri.startswith("gs://"):
                                         signed_video_url = generate_signed_url(vid_uri, expiration_minutes=10)
                                         if signed_video_url:
                                             st.video(signed_video_url)
                                             # st.caption(f"Clip loaded via temporary URL.") # Optional caption
                                         else:
                                             st.warning(f"No URL for {os.path.basename(vid_uri)}")
                                     else:
                                          st.warning("No valid GCS URI.")
                                     # st.divider()
                                 col_idx_vid += 1
                    else:
                        st.info("Intermediate video clip data is missing or not a list.")

                # --- Optionally display raw final state for debugging ---
                with st.expander("Show Raw Final State (for Debugging)"):
                    # Exclude potentially very large fields for readability
                    debug_state = {k: v for k, v in final_agent_state.items() if k not in [
                        'structured_data', 'narrative_context', 'draft', 'player_lookup_dict',
                        'generated_content', 'all_image_assets', 'all_video_assets',
                        'generated_visual_assets', 'generated_video_assets', 'word_timestamps'
                        ]}
                    try:
                        # Attempt to pretty-print JSON
                        st.json(json.dumps(debug_state, indent=2, default=str))
                    except Exception:
                        # Fallback to plain text if JSON fails (e.g., non-serializable types)
                        st.text(str(debug_state))

            # Display message if the run completed but no final state was captured (e.g., early error not caught)
            elif not st.session_state.get("error_message"):
                st.warning("Agent run completed, but no final state was captured. Check application logs for potential errors during the agent execution.")

    # Message shown before any run is attempted
    elif not st.session_state.run_complete and not st.session_state.error_message:
        with results_container:
            st.info("Select a team and click the 'Generate Video' button in the left panel to start.")

# --- Footer or additional info ---
st.sidebar.markdown("---")
st.sidebar.markdown("Powered by LangGraph & Google Cloud Vertex AI")
