# mlb_agent_ui.py
import streamlit as st
import json
import logging
from datetime import datetime, timedelta, UTC # Added timedelta
import time
from typing import Optional, Dict, Any, List # Added for type hints

# --- GCS Client & Secret Manager ---
from google.cloud import storage
from google.api_core import exceptions as google_exceptions
from google.cloud import secretmanager
from google.oauth2 import service_account

# --- Firestore Client --- NEW
from google.cloud import firestore
from google.api_core.exceptions import NotFound as FirestoreNotFound

# --- Import necessary components from your agent script ---
# Assume mlb_agent5.py is in the same directory or accessible in PYTHONPATH
try:
    from mlb_agent5 import (
        app,  # The compiled LangGraph app
        TEAMS,
        get_latest_final_game_pk,
        load_player_metadata,
        # load_player_metadata, # Defined below for clarity
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
SERVICE_ACCOUNT_SECRET_ID = "streamlit-gcs-sa-key" # <-- **REPLACE** with your Secret ID for GCS Signed URLs
SECRET_VERSION = "latest"
# Ensure GCP_PROJECT_ID is available (either imported or defined above)
if 'GCP_PROJECT_ID' not in globals():
     st.error("GCP_PROJECT_ID is not defined. Please define it or import from agent.")
     st.stop()

FIRESTORE_COLLECTION = "mlb_agent_access" # Name of your Firestore collection for access control

# --- Initialize Firestore Client --- NEW
@st.cache_resource
def initialize_firestore_client(project_id: str) -> Optional[firestore.Client]:
    """Initializes the Firestore client using Application Default Credentials."""
    try:
        db = firestore.Client(project=project_id)
        logger.info("Successfully initialized Firestore client.")
        return db
    except Exception as e:
        logger.error(f"Failed to initialize Firestore client: {e}", exc_info=True)
        st.error(f"⚠️ Could not connect to Firestore for access control: {e}")
        return None

firestore_db = initialize_firestore_client(GCP_PROJECT_ID)

# --- Access Control Functions --- NEW

def check_user_access(db: firestore.Client, user_id: str) -> str:
    """Checks user access status in Firestore.
    Returns: 'approved', 'requested', 'denied', 'not_found'
    """
    if not db or not user_id:
        return "error_db_or_id" # Indicate DB init failure or missing ID

    try:
        doc_ref = db.collection(FIRESTORE_COLLECTION).document(user_id)
        doc = doc_ref.get()
        if doc.exists:
            data = doc.to_dict()
            status = data.get("status", "not_found").lower()
            # Allow for legacy 'granted' status as well
            if status in ["approved", "granted"]:
                logger.info(f"Access check for '{user_id}': Approved")
                return "approved"
            elif status == "requested":
                logger.info(f"Access check for '{user_id}': Requested")
                return "requested"
            elif status == "denied":
                 logger.info(f"Access check for '{user_id}': Denied")
                 return "denied"
            else:
                logger.warning(f"Access check for '{user_id}': Found doc but unknown status '{status}'")
                return "not_found" # Treat unknown status as not found for access purposes
        else:
            logger.info(f"Access check for '{user_id}': Not Found")
            return "not_found"
    except Exception as e:
        logger.error(f"Error checking Firestore access for '{user_id}': {e}", exc_info=True)
        return "error_firestore_check"

def request_access(db: firestore.Client, user_id: str) -> bool:
    """Records an access request in Firestore."""
    if not db or not user_id:
        logger.error("Firestore DB not initialized or user_id missing for request.")
        return False
    try:
        doc_ref = db.collection(FIRESTORE_COLLECTION).document(user_id)
        doc_ref.set({
            "status": "requested",
            "request_timestamp": firestore.SERVER_TIMESTAMP, # Record request time
            "user_id": user_id # Store the user ID explicitly
        }, merge=True) # Use merge=True to avoid overwriting if approved/denied later
        logger.info(f"Access request recorded for '{user_id}'.")
        return True
    except Exception as e:
        logger.error(f"Error recording access request for '{user_id}': {e}", exc_info=True)
        return False

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

        credentials = service_account.Credentials.from_service_account_info(
            sa_key_dict,
            scopes=['https://www.googleapis.com/auth/devstorage.read_only'] # Scope just needed to read objects
        )
        logger.info(f"Successfully loaded signing credentials from secret: {secret_id}")
        return credentials
    except Exception as e:
        logger.error(f"Failed to load signing credentials from Secret Manager ({secret_id}): {e}", exc_info=True)
        # Display error later if needed, avoid stopping the app here
        # st.error(f"⚠️ Failed to load credentials from Secret Manager ({secret_id}) required for media display. Ensure Secret exists and Cloud Run SA has Secret Accessor role.")
        return None

# --- Initialize GCS Client & Load Credentials ---
# Load credentials first, check availability later before generating URLs
gcs_signing_credentials = load_gcs_signing_credentials(GCP_PROJECT_ID, SERVICE_ACCOUNT_SECRET_ID, SECRET_VERSION)

try:
    storage_client = storage.Client(project=GCP_PROJECT_ID) # Use ADC or specify credentials if needed globally
    logger.info("Initialized Google Cloud Storage client.")
except Exception as e:
    st.error(f"⚠️ Failed to initialize Google Cloud Storage client: {e}. Media display might fail.")
    storage_client = None
    logger.error(f"Failed to initialize GCS client: {e}", exc_info=True)


# --- Streamlit App Configuration ---
st.title("⚾ MLB Game Recap Generator")

# --- Session State Initialization ---
if 'run_complete' not in st.session_state:
    st.session_state.run_complete = False
if 'final_state' not in st.session_state: # Changed from final_result
    st.session_state.final_state = None
if 'error_message' not in st.session_state:
    st.session_state.error_message = None
if 'selected_team_key' not in st.session_state:
     st.session_state.selected_team_key = list(TEAMS.keys())[0] # Default selection
if 'user_id' not in st.session_state: # For storing the identified user
    st.session_state.user_id = None
if 'access_status' not in st.session_state: # 'approved', 'requested', 'denied', 'not_found', 'error_*'
    st.session_state.access_status = None

# --- Helper Function for Signed URLs ---
def generate_signed_url(gcs_uri: str, expiration_minutes: int = 30) -> Optional[str]:
    """Generates a temporary signed URL for a GCS object."""
    global storage_client, gcs_signing_credentials # Use globally defined vars
    if not storage_client:
        logger.error("GCS storage client not available for generating signed URL.")
        return None
    if not gcs_signing_credentials: # Check if credentials loaded successfully
        logger.error("Signing credentials not available for generating signed URL.")
        st.error("⚠️ Cannot generate media URLs: Signing credentials failed to load.", icon="🔒") # Show error here if needed
        return None
    if not gcs_uri or not gcs_uri.startswith("gs://"):
        logger.warning(f"Invalid GCS URI provided for signing: {gcs_uri}")
        return None

    try:
        bucket_name, blob_name = gcs_uri.replace("gs://", "").split("/", 1)
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        signed_url = blob.generate_signed_url(
            version="v4",
            expiration=timedelta(minutes=expiration_minutes),
            method="GET",
            credentials=gcs_signing_credentials,
        )
        logger.debug(f"Generated signed URL for: {gcs_uri}")
        return signed_url
    except google_exceptions.NotFound:
         logger.error(f"Blob not found when trying to generate signed URL: {gcs_uri}")
         return None
    except google_exceptions.Forbidden as e:
         logger.error(f"Permission denied generating signed URL for {gcs_uri}. Check Service Account Token Creator role ON SA and Storage Object Viewer role ON bucket. Error: {e}", exc_info=False)
         st.warning(f"Permission error generating URL for {gcs_uri}. Check app logs and IAM roles.", icon="🔒")
         return None
    except Exception as e:
        logger.error(f"Error generating signed URL for {gcs_uri}: {e}", exc_info=True)
        return None

# --- Agent Streaming Function ---
# (No changes needed in this function itself)
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

    node_emojis = {
        "planner": "📅", "retrieve_data": "🔍", "generate": "✍️",
        "reflect": "🤔", "research_critique": "📚", "web_search_context": "🌐",
        "analyze_script_for_images": "🖼️", "retrieve_images": "🏞️",
        "analyze_script_for_visual_prompts": "🎬", "generate_visuals": "✨",
        "critique_visuals": "🧐", "generate_video_clips": "🎥",
        "aggregate_final_output": "📦", "generate_audio": "🔊", "__end__": "🏁"
    }

    try:
        recursion_limit = 50 # Or get from config/state if needed
        logger.info("Starting agent stream...")
        # Use app.stream which yields updates
        stream = agent_app.stream(initial_agent_state, {"recursion_limit": recursion_limit}, stream_mode="updates")

        yield "🚀 **Agent execution started...**\n\n" # Initial message

        for step_update in stream:
            # step_update is a dictionary where keys are nodes that just ran
            # The value associated with the key is the *entire current state*
            if not step_update: # Handle potential empty updates if graph yields them
                continue
            node_name = list(step_update.keys())[0]
            nodes_seen_in_stream.append(node_name) # <-- Log node name
            latest_full_state = step_update[node_name] # Capture the full state dictionary
            logger.info(f"STREAM UI: Received update from node: {node_name}") # <-- Add server log

            emoji = node_emojis.get(node_name, "⚙️") # Default emoji
            # Handle potential errors within the state update itself
            node_error = latest_full_state.get("error") if isinstance(latest_full_state, dict) else None

            status_message = f"{emoji} **{node_name.replace('_', ' ').title()}**"
            if node_error:
                 status_message += f" completed with error: {str(node_error)[:100]}..." # Show partial error
                 logger.warning(f"Node '{node_name}' reported error in stream: {node_error}")
                 # Optionally set the main error message if a node reports one
                 st.session_state['error_message'] = f"Error in node '{node_name}': {node_error}"
            else:
                 status_message += " finished."


            # Optional: Add specific details for key nodes based on state
            if isinstance(latest_full_state, dict): # Check if state is a dictionary
                 if node_name == "generate":
                     status_message += f" (Script Revision {latest_full_state.get('revision_number', '?')})"
                 elif node_name == "generate_visuals":
                     vis_rev = latest_full_state.get('visual_revision_number', '?')
                     # Note: generate_visuals node increments *before* returning, so the number
                     # shown will be the revision *about to be generated* or *just finished*
                     # depending on interpretation. Let's show the number associated with the finished state.
                     status_message += f" (Visual Revision {vis_rev -1 if isinstance(vis_rev, int) and vis_rev > 0 else vis_rev})" # Show completed rev
                 elif node_name == "generate_audio":
                     if latest_full_state.get("generated_audio_uri"):
                         status_message += " Audio generated."
                 # --- CORRECTED LENGTH CHECKS ---
                 elif node_name == "retrieve_data":
                      narrative_ctx = latest_full_state.get('narrative_context') # Get value, could be None
                      ctx_len = len(narrative_ctx) if isinstance(narrative_ctx, list) else 0 # Check type before len()

                      struct_data = latest_full_state.get('structured_data') # Get value
                      struct_len = len(struct_data) if isinstance(struct_data, list) else 0 # Check type before len()
                      status_message += f" (Context: {ctx_len}, Structured: {struct_len})"

                 elif node_name == "retrieve_images":
                      retrieved_images = latest_full_state.get('retrieved_image_data')
                      img_len = len(retrieved_images) if isinstance(retrieved_images, list) else 0
                      status_message += f" (Static Assets: {img_len})"

                 elif node_name == "generate_visuals":
                      gen_visuals = latest_full_state.get('generated_visual_assets')
                      vis_len = len(gen_visuals) if isinstance(gen_visuals, list) else 0
                      status_message += f" (Generated Images: {vis_len})" # Note: This shows count *after* generation

                 elif node_name == "generate_video_clips":
                      gen_videos = latest_full_state.get('generated_video_assets')
                      vid_len = len(gen_videos) if isinstance(gen_videos, list) else 0
                      status_message += f" (Generated Videos: {vid_len})"
                 # --- END CORRECTED LENGTH CHECKS ---

            yield status_message + "\n\n" # Add extra newline for spacing
            # time.sleep(0.1) # Optional small delay

        # After the loop finishes, store the VERY LAST complete state captured
        final_state_on_success = latest_full_state
        st.session_state['final_state'] = final_state_on_success
        logger.info(f"STREAM UI: Stream finished. Nodes seen: {nodes_seen_in_stream}") # <-- Log all nodes seen
        logger.info(f"STREAM UI: Storing final state with keys: {list(st.session_state['final_state'].keys()) if st.session_state['final_state'] else 'None'}")
        yield f"\n{node_emojis['__end__']} **Agent execution complete! Processing results...**"

    except Exception as e:
        # ---- If an exception occurs DURING the stream ----
        logger.error(f"STREAM UI: Error during streaming: {e}", exc_info=True)
        logger.warning(f"STREAM UI: Stream errored. Nodes seen before error: {nodes_seen_in_stream}") # <-- Log nodes seen before error
        st.session_state['error_message'] = f"Critical Error during stream: {e}"
        st.session_state['final_state'] = latest_full_state # Store partial state
        logger.warning(f"STREAM UI: Storing partial state on error with keys: {list(st.session_state['final_state'].keys()) if st.session_state['final_state'] else 'None'}")
        yield f"\n❌ **{st.session_state['error_message']}**"
        # We still mark run complete here, but the error will be shown
        st.session_state.run_complete = True
        st.rerun() # Rerun to ensure UI reflects the error state properly


# --- Player Metadata Loading Function (Example - Adapt if defined elsewhere) ---
# This version uses the imported components if available
# @st.cache_data # Cache the lookup dict for the session
def load_player_metadata() -> Dict[int, str]:
    """Loads player ID to name mapping from BQ."""
    logger.info("Loading player metadata via Streamlit UI...")
    # Use BQ client initialized in the agent script (ensure it's accessible)
    from mlb_agent5 import execute_bq_query, GCP_PROJECT_ID, BQ_DATASET_ID, PLAYER_METADATA_TABLE_ID
    import pandas as pd
    player_lookup_dict = {}
    try:
        player_lookup_query = f"SELECT player_id, player_name FROM `{GCP_PROJECT_ID}.{BQ_DATASET_ID}.{PLAYER_METADATA_TABLE_ID}`"
        player_results_df = execute_bq_query(player_lookup_query)

        if player_results_df is not None and not player_results_df.empty:
            # Ensure player_id is treated as integer, handle potential NaN/None before conversion
            player_lookup_dict = {
                int(row['player_id']): row['player_name']
                for index, row in player_results_df.iterrows()
                if pd.notna(row['player_id']) and pd.notna(row['player_name'])
            }
            logger.info(f"Loaded {len(player_lookup_dict)} player names into lookup dictionary.")
        else:
            logger.warning("Player metadata query failed or returned no results. Lookup dictionary is empty.")
            player_lookup_dict = {} # Ensure it's an empty dict on failure
    except Exception as meta_err:
         logger.error(f"Failed to load player metadata: {meta_err}. Proceeding with empty lookup.", exc_info=True)
         player_lookup_dict = {}
    return player_lookup_dict

# ==============================================================================
# --- MAIN APPLICATION LOGIC with Access Control ---
# ==============================================================================

# 1. Identify User & Check Access (or Request Access)
if not st.session_state.user_id or not st.session_state.access_status:
    st.subheader("Welcome!")
    st.markdown("Please enter your email address to check your access status or request access.")

    # --- !! WARNING: Simple Email Input - Not Secure Authentication !! ---
    # In production, replace this with a real authentication method
    # (e.g., reading from st.user.email on Streamlit Cloud, using IAP headers, OAuth)
    email_input = st.text_input("Your Email Address:", key="email_input", help="Enter the email you use to access this application.")
    placeholder = st.empty() # Placeholder for buttons/status

    if email_input:
        # Basic email format check (not exhaustive)
        if "@" not in email_input or "." not in email_input:
            placeholder.warning("Please enter a valid email address.")
        else:
            user_id = email_input.lower().strip() # Normalize email
            st.session_state.user_id = user_id # Store identified user

            # Check access status only if Firestore client is available
            if firestore_db:
                status = check_user_access(firestore_db, user_id)
                st.session_state.access_status = status

                if status == "approved":
                    placeholder.success(f"Access approved for {user_id}. Loading application...")
                    time.sleep(1) # Brief pause before rerun
                    st.rerun()
                elif status == "requested":
                    placeholder.info(f"Access request for {user_id} is pending approval. Please check back later.")
                    # Keep showing this message, hide the request button
                elif status == "denied":
                    placeholder.error(f"Access has been denied for {user_id}. Please contact the administrator if you believe this is an error.")
                elif status == "not_found":
                    placeholder.warning(f"No access record found for {user_id}.")
                    if placeholder.button("Request Access", key="request_access_button"):
                        if request_access(firestore_db, user_id):
                            st.session_state.access_status = "requested" # Update state
                            st.success("Access request submitted! You will be notified once approved.")
                            time.sleep(2)
                            st.rerun()
                        else:
                            st.error("Failed to submit access request. Please try again later or contact support.")
                elif status == "error_firestore_check":
                     placeholder.error("Could not check access status due to a Firestore error. Please try again later.")
                elif status == "error_db_or_id":
                     placeholder.error("Internal error: Firestore client not ready or User ID missing.")
            else:
                # Firestore client failed to initialize
                placeholder.error("Access control system is currently unavailable. Cannot verify or request access.")
                # Stop execution here as access cannot be determined
                st.stop()

# 2. If Access Approved, Show the Main Application
elif st.session_state.access_status == "approved":

    st.markdown(f"Welcome, **{st.session_state.user_id}**!") # Acknowledge the user
    st.markdown("Select a team to generate a recap of their latest completed game, including script, images, video clips, and audio.")

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
        on_change=lambda: st.session_state.update(run_complete=False, final_state=None, error_message=None) # Reset on change
    )

    st.session_state.selected_team_key = team_key_lookup[selected_team_display_name]
    selected_team_id = TEAMS[st.session_state.selected_team_key]
    st.write(f"Selected Team: **{selected_team_display_name}** (ID: {selected_team_id})")


    # --- Run Button and Execution Logic ---
    if st.button(f"Generate Recap for {selected_team_display_name}'s Latest Game", key="run_button"):
        # Reset state for a new run
        st.session_state.run_complete = False
        st.session_state.final_state = None
        st.session_state.error_message = None

        # Create a container for the streaming output
        stream_output_container = st.container(border=True) # Add border for visibility

        # Check if signing credentials loaded - crucial for displaying results later
        if not gcs_signing_credentials:
            st.session_state.error_message = "Failed to load necessary credentials from Secret Manager. Cannot generate signed URLs for media."
            st.session_state.run_complete = True # Mark as complete to show error
            st.rerun() # Rerun to display the error message
        else:
            try:
                # 1. Find Latest Game PK
                with stream_output_container: st.info("Finding latest completed game...")
                logger.info(f"Attempting to find latest game PK for team ID: {selected_team_id}")
                latest_game_pk = get_latest_final_game_pk(selected_team_id)
                if not latest_game_pk:
                    st.session_state.error_message = f"❌ Could not find a recent completed game for {selected_team_display_name}."
                    st.session_state.run_complete = True
                    st.rerun()

                # 2. Load Player Metadata
                with stream_output_container: st.info("Loading player metadata...")
                try:
                    player_lookup = load_player_metadata() # Load fresh data
                    if not player_lookup:
                        with stream_output_container: st.warning("⚠️ Could not load player metadata.")
                    else:
                        logger.info(f"Successfully loaded {len(player_lookup)} players.")
                except Exception as load_err:
                    logger.error(f"Error loading player metadata: {load_err}", exc_info=True)
                    with stream_output_container: st.error("Failed to load player metadata.")
                    player_lookup = {} # Ensure it's an empty dict

                with stream_output_container: st.info(f"Found latest game (PK: {latest_game_pk}). Preparing agent...")

                # 3. Prepare Initial State
                task = f"Provide a detailed recap of game {latest_game_pk} ({selected_team_display_name}), highlighting impactful plays and player performances in an engaging two-host dialogue format."
                initial_state = {
                    "task": task, "game_pk": latest_game_pk, "max_revisions": 2, "revision_number": 0,
                    "player_lookup_dict": player_lookup, # Use freshly loaded data
                    "narrative_context": [], "all_image_assets": [],
                    "all_video_assets": [], "generated_visual_assets": [], "generated_video_assets": [],
                    "visual_revision_number": 0, "max_visual_revisions": 2, "error": None, "plan": None,
                    "structured_data": None, "image_search_queries":None, "retrieved_image_data":None,
                    "draft": None, "critique": None, "generated_content": None, "visual_generation_prompts": [],
                    "visual_critique": None, "generated_audio_uri": None,
                }
                logger.info(f"Initial state prepared for game PK {latest_game_pk}.")

                # 4. Execute Agent via Streaming
                start_time = time.time()
                with stream_output_container:
                    # run_agent_and_stream_progress handles setting final_state/error_message
                    st.write_stream(run_agent_and_stream_progress(app, initial_state))
                end_time = time.time()
                logger.info(f"Agent streaming process finished UI-side in {end_time - start_time:.2f} seconds.")

            except Exception as e:
                logger.error(f"An unexpected error occurred *before* agent streaming started: {e}", exc_info=True)
                st.session_state.error_message = f"❌ An unexpected error occurred before agent start: {e}"
            finally:
                # Ensure run_complete is set so results/errors are displayed after rerun
                st.session_state.run_complete = True
                st.rerun() # Rerun to display results or errors handled within the try/except/stream

    # --- Display Results ---
    if st.session_state.run_complete:
        # Check for errors FIRST (set by stream function or outer try/except)
        if st.session_state.get("error_message"):
            st.error(st.session_state.error_message)

        # Check if we have a final state dictionary to process
        final_agent_state = st.session_state.get('final_state') # Use .get() for safety
        if final_agent_state and isinstance(final_agent_state, dict):
            st.success(f"Recap Generation Process Complete for Game PK: {final_agent_state.get('game_pk', 'N/A')}")

            # --- Display Generated Script ---
            st.subheader("🎙️ Generated Dialogue Script")
            script_content = final_agent_state.get("generated_content") # Read from final_agent_state
            if script_content:
                script_lines = script_content.strip().split('\n')
                display_script = ""
                for i, line in enumerate(script_lines):
                     if line.strip():
                         # Basic alternating speaker assignment
                         speaker = "Host 1:" if i % 2 == 0 else "Host 2:"
                         display_script += f"**{speaker}** {line}\n\n"
                st.markdown(display_script)
            else:
                st.warning("No final script content was generated.")

            # --- Display Generated Audio ---
            st.subheader("🎧 Generated Audio")
            audio_uri = final_agent_state.get("generated_audio_uri") # Read from final_agent_state
            if audio_uri:
                st.write(f"Audio GCS URI: `{audio_uri}`")
                if isinstance(audio_uri, str) and audio_uri.startswith("gs://"):
                    signed_audio_url = generate_signed_url(audio_uri)
                    if signed_audio_url:
                        try:
                            st.audio(signed_audio_url, format='audio/mp3') # Specify format if known
                            st.caption(f"Audio player loaded via temporary URL (expires).")
                        except Exception as audio_err:
                            st.warning(f"Could not display audio player for {audio_uri}. Error: {audio_err}")
                            st.markdown(f"Direct Link (expires): [Listen to Audio]({signed_audio_url})")
                    else:
                        st.warning(f"Could not generate temporary URL to play audio from: {audio_uri}")
                else:
                    st.warning(f"Invalid or non-GCS audio URI format: {audio_uri}")
            else:
                st.warning("No audio URI found in the final state.")

            # --- Display Visual Assets (Images) ---
            st.subheader("🖼️ Visual Assets (Images)")
            all_image_assets = final_agent_state.get("all_image_assets") # Read from final_agent_state
            if all_image_assets and isinstance(all_image_assets, list):
                if not all_image_assets:
                    st.info("No image assets were found in the final state.")
                else:
                    num_image_cols = 4 # Adjust as needed
                    cols = st.columns(num_image_cols)
                    col_idx = 0
                    for i, asset in enumerate(all_image_assets):
                        with cols[col_idx % len(cols)]:
                            st.markdown(f"**Image {i+1}**")
                            asset_type = asset.get("type", asset.get("image_type", "N/A")).replace("_", " ").title() # More robust type getting
                            model = asset.get("model_used", "")
                            source_info = ""
                            if asset_type == "Generated Image":
                                prompt_origin = asset.get('prompt_origin', 'N/A')
                                source_info = f"Generated ({model}) from prompt: *'{prompt_origin[:50]}{'...' if len(prompt_origin) > 50 else ''}'*"
                            elif asset_type == "Headshot":
                                source_info = f"Static Headshot for: *{asset.get('entity_name', 'N/A')}*"
                            elif asset_type == "Logo":
                                source_info = f"Static Logo Search: *'{asset.get('search_term_origin', 'N/A')}'*"
                            else:
                                source_info = f"Type: {asset_type}"

                            st.markdown(f"_{source_info}_")
                            img_uri = asset.get("image_uri")
                            st.write(f"GCS URI: `{img_uri}`")

                            if img_uri and isinstance(img_uri, str) and img_uri.startswith("gs://"):
                                signed_image_url = generate_signed_url(img_uri)
                                if signed_image_url:
                                    try:
                                        st.image(signed_image_url, use_column_width=True) # Adjust width
                                        st.caption(f"Image loaded via temporary URL (expires).")
                                    except Exception as img_err:
                                        st.warning(f"Could not display image for {img_uri}. Error: {img_err}")
                                        st.markdown(f"Direct Link (expires): [View Image]({signed_image_url})")
                                else:
                                    st.warning(f"Could not generate temporary URL for image: {img_uri}")
                            elif img_uri:
                                st.warning(f"Image URI is not a GCS URI, attempting direct display: {img_uri}")
                                try: st.image(img_uri, use_column_width=True)
                                except: st.error(f"Failed to display non-GCS image URI.")
                            else:
                                st.warning("Image URI missing.")
                            # st.divider() # Maybe remove divider inside columns
                        col_idx += 1
            else:
                st.warning("Image assets data is missing or not a list in the final state.")

            # --- Display Visual Assets (Videos) ---
            st.subheader("🎬 Visual Assets (Videos)")
            all_video_assets = final_agent_state.get("all_video_assets") # Read from final_agent_state
            if all_video_assets and isinstance(all_video_assets, list):
                if not all_video_assets:
                    st.info("No video assets were found in the final state.")
                else:
                    num_video_cols = 3 # Adjust as needed
                    cols_vid = st.columns(num_video_cols)
                    col_idx_vid = 0
                    for i, asset in enumerate(all_video_assets):
                        with cols_vid[col_idx_vid % len(cols_vid)]:
                            st.markdown(f"**Video Clip {i+1}**")
                            model = asset.get("model_used", "N/A")
                            prompt = asset.get("source_prompt", 'N/A')
                            prompt_display = f"'{prompt[:60]}{'...' if len(prompt) > 60 else ''}'"
                            st.markdown(f"Generated ({model}) from prompt: *{prompt_display}*")

                            vid_uri = asset.get("video_uri")
                            st.write(f"GCS URI: `{vid_uri}`")
                            if vid_uri and isinstance(vid_uri, str) and vid_uri.startswith("gs://"):
                                signed_video_url = generate_signed_url(vid_uri)
                                if signed_video_url:
                                    try:
                                        st.video(signed_video_url, format='video/mp4') # Specify format if known
                                        st.caption(f"Video loaded via temporary URL (expires).")
                                    except Exception as vid_err:
                                        st.warning(f"Could not display video player for {vid_uri}. Error: {vid_err}")
                                        st.markdown(f"Direct Link (expires): [View Video]({signed_video_url})")
                                else:
                                    st.warning(f"Could not generate temporary URL for video: {vid_uri}")
                            elif vid_uri:
                                st.warning(f"Video URI is not a GCS URI, attempting direct display: {vid_uri}")
                                try: st.video(vid_uri)
                                except: st.error("Failed to display non-GCS video URI.")
                            else:
                                st.warning("Video URI missing.")
                            # st.divider() # Maybe remove divider inside columns
                        col_idx_vid += 1
            else:
                st.warning("Video assets data is missing or not a list in the final state.")

            # --- Optionally display raw final state for debugging ---
            with st.expander("Show Raw Final State (for Debugging)"):
                # Filter out large/complex objects for readability
                debug_state = {k: v for k, v in final_agent_state.items() if k not in [
                    'structured_data', 'narrative_context', 'draft',
                    'generated_content', 'all_image_assets', 'all_video_assets',
                    'generated_visual_assets', 'generated_video_assets',
                    'player_lookup_dict', 'retrieved_image_data', 'image_search_queries',
                    'visual_generation_prompts'
                    ]}
                try:
                    st.json(json.dumps(debug_state, indent=2, default=str)) # Use json.dumps for better handling
                except Exception as json_err:
                    st.warning(f"Could not serialize debug state to JSON: {json_err}")
                    st.text(str(debug_state)) # Fallback to string representation

        # Display message if the run completed but no final state was captured (e.g., early error not caught by message)
        elif not st.session_state.get("error_message"):
            st.warning("Agent run completed, but no final state was captured. Check logs for details.")

    # Message shown before any run is attempted *by an approved user*
    elif not st.session_state.run_complete and not st.session_state.error_message:
        st.info("Select a team and click the button above to generate the game recap.")


# 3. If Access Requested or Denied
elif st.session_state.access_status == "requested":
    st.info(f"Access request for **{st.session_state.user_id}** is pending approval. Please check back later or contact the administrator.")
    # Optionally add a refresh button
    if st.button("Check Status Again"):
        st.session_state.access_status = None # Force re-check on rerun
        st.rerun()

elif st.session_state.access_status == "denied":
    st.error(f"Access has been denied for **{st.session_state.user_id}**. Please contact the administrator if you believe this is an error.")

# 4. Handle Firestore Errors during initial check phase (if status is error_*)
elif st.session_state.access_status and st.session_state.access_status.startswith("error_"):
     st.error(f"A system error occurred while checking access ({st.session_state.access_status}). Please try refreshing the page or contact support.")
     if st.button("Retry"):
         st.session_state.access_status = None
         st.session_state.user_id = None # Reset user ID as well
         st.rerun()

# --- Footer or additional info ---
st.markdown("---")
st.markdown("Powered by LangGraph, Vertex AI (Gemini, Imagen, Veo, TTS), BigQuery, GCS, and Firestore.")
st.caption(f"GCP Project: {GCP_PROJECT_ID}")
if st.session_state.user_id:
    st.caption(f"User ID: {st.session_state.user_id} | Access Status: {st.session_state.access_status}")