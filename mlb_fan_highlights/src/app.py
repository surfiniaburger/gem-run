# mlb_agent_ui.py
import streamlit as st
import json
import logging
from datetime import datetime, timedelta, UTC # Added timedelta
import time
from typing import Optional, Dict, Any, List # Added for type hints

# --- GCS Client ---
from google.cloud import storage
from google.api_core import exceptions as google_exceptions

# --- Import necessary components from your agent script ---
# Assume mlb_agent5.py is in the same directory or accessible in PYTHONPATH
try:
    from mlb_agent5 import (
        app,  # The compiled LangGraph app
        TEAMS,
        get_latest_final_game_pk,
        load_player_metadata,
        # Define load_player_metadata if it exists in your agent script,
        # otherwise load it here or remove if not needed for initial state
        # load_player_metadata, # Example import if defined in agent
        logger, # Use the same logger setup if desired
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

# --- Streamlit App Configuration ---
st.set_page_config(page_title="MLB Game Recap Generator", layout="wide")
st.title("⚾ MLB Game Recap Generator")
st.markdown("Select a team to generate a recap of their latest completed game, including script, images, video clips, and audio.")

# --- Initialize GCS Client ---
try:
    storage_client = storage.Client()
    logger.info("Initialized Google Cloud Storage client for Signed URLs.")
except Exception as e:
    st.error(f"⚠️ Failed to initialize Google Cloud Storage client: {e}. Media display might fail. Ensure Application Default Credentials are set.")
    storage_client = None # Set to None so checks below fail gracefully
    logger.error(f"Failed to initialize GCS client: {e}", exc_info=True)

# --- Session State Initialization ---
if 'run_complete' not in st.session_state:
    st.session_state.run_complete = False
if 'final_state' not in st.session_state: # Changed from final_result
    st.session_state.final_state = None
if 'error_message' not in st.session_state:
    st.session_state.error_message = None
if 'selected_team_key' not in st.session_state:
     st.session_state.selected_team_key = list(TEAMS.keys())[0] # Default selection

# --- Helper Function for Signed URLs ---
def generate_signed_url(gcs_uri: str, expiration_minutes: int = 30) -> Optional[str]:
    """Generates a temporary signed URL for a GCS object."""
    global storage_client
    if not storage_client:
        logger.error("GCS storage client not available for generating signed URL.")
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
def run_agent_and_stream_progress(agent_app, initial_agent_state):
    """
    Runs the agent using app.stream() and yields formatted progress strings.
    Stores the FINAL complete state in st.session_state.
    """
    latest_full_state = None # Variable to hold the most recent complete state dictionary
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
            latest_full_state = step_update[node_name] # Capture the full state dictionary

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
        # Only store if the stream didn't end abruptly with an exception below
        st.session_state['final_state'] = latest_full_state
        logger.info("Agent stream finished successfully.")
        yield f"\n{node_emojis['__end__']} **Agent execution complete! Processing results...**"

    except Exception as e:
        logger.error(f"Error during agent streaming execution: {e}", exc_info=True)
        st.session_state['error_message'] = f"Error during agent execution: {e}" # Store the exception message
        yield f"\n❌ **Critical Error during stream:** {e}"
        st.session_state['final_state'] = latest_full_state # Store partial state on error if available
        # Make sure error state persists for display after rerun
        st.session_state.run_complete = True # Mark as complete even on error to show message
        st.rerun() # Rerun to display the error message now

# --- Player Metadata Loading Function (Example - Adapt if defined elsewhere) ---
# If load_player_metadata is NOT in mlb_agent5.py, define it here.
# If it IS in mlb_agent5.py, remove this definition and import it.
def load_player_metadata() -> Dict[int, str]:
    """Loads player ID to name mapping from BQ."""
    logger.info("Loading player metadata via Streamlit UI...")
    # Use BQ client initialized in the agent script (ensure it's accessible)
    # Or re-initialize a BQ client here if needed
    from mlb_agent5 import execute_bq_query, GCP_PROJECT_ID, BQ_DATASET_ID, PLAYER_METADATA_TABLE_ID
    import pandas as pd
    player_lookup_dict = {}
    try:
        player_lookup_query = f"SELECT player_id, player_name FROM `{GCP_PROJECT_ID}.{BQ_DATASET_ID}.{PLAYER_METADATA_TABLE_ID}`"
        player_results_df = execute_bq_query(player_lookup_query)

        if player_results_df is not None and not player_results_df.empty:
            player_lookup_dict = {int(row['player_id']): row['player_name'] for index, row in player_results_df.iterrows() if pd.notna(row['player_id'])}
            logger.info(f"Loaded {len(player_lookup_dict)} player names into lookup dictionary.")
        else:
            logger.warning("Player metadata query failed or returned no results. Lookup dictionary is empty.")
    except Exception as meta_err:
         logger.error(f"Failed to load player metadata: {meta_err}. Proceeding with empty lookup.", exc_info=True)
         player_lookup_dict = {}
    return player_lookup_dict

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


# --- UPDATED Run Button and Execution Logic ---
if st.button(f"Generate Recap for {selected_team_display_name}'s Latest Game", key="run_button"):
    # Reset state for a new run
    st.session_state.run_complete = False
    st.session_state.final_state = None  # Use 'final_state' now
    st.session_state.error_message = None

    # Create a container for the streaming output
    stream_output_container = st.container(border=True) # Add border for visibility

    try:
        # 1. Find Latest Game PK (Keep this part outside the stream)
        with stream_output_container:
            st.info("Finding latest completed game...")
        logger.info(f"Attempting to find latest game PK for team ID: {selected_team_id}")
        latest_game_pk = get_latest_final_game_pk(selected_team_id)

        if not latest_game_pk:
            logger.error(f"Could not find a recent completed game for team {selected_team_display_name} (ID: {selected_team_id}).")
            st.session_state.error_message = f"❌ Could not find a recent completed game for {selected_team_display_name}. Please try another team or check game schedules."
            st.session_state.run_complete = True
            st.rerun() # Rerun to display error message immediately

        # 2. Load Player Metadata (Keep outside stream, happens once)
        with stream_output_container:
            st.info("Loading player metadata...")
        try:
             player_lookup = load_player_metadata() # Call the function defined above or imported
             if not player_lookup:
                 logger.warning("Player metadata lookup returned empty.")
                 with stream_output_container:
                     st.warning("⚠️ Could not load player metadata. Headshot images may be unavailable.")
             else:
                 logger.info(f"Successfully loaded {len(player_lookup)} players via Streamlit.")
        except Exception as load_err:
             logger.error(f"Error calling load_player_metadata from Streamlit: {load_err}", exc_info=True)
             with stream_output_container:
                st.error("Failed to load player metadata. Headshots may be unavailable.")
             player_lookup = {} # Ensure it's an empty dict on error

        logger.info(f"Found latest game PK: {latest_game_pk} for team {selected_team_display_name}")
        with stream_output_container:
            st.info(f"Found latest game (PK: {latest_game_pk}). Preparing agent...")

        # 3. Prepare Initial State (Keep this part outside the stream)
        task = f"Provide a detailed recap of game {latest_game_pk} ({selected_team_display_name}), highlighting impactful plays and player performances in an engaging two-host dialogue format."
        initial_state = {
            "task": task,
            "game_pk": latest_game_pk,
            "max_revisions": 2,
            "revision_number": 0,
            "player_lookup_dict": player_lookup, # Pass loaded metadata
            "narrative_context": [],
            "all_image_assets": [],
            "all_video_assets": [],
            "generated_visual_assets": [],
            "generated_video_assets": [],
            "visual_revision_number": 0,
            "max_visual_revisions": 2,
            "error": None,
            "plan": None,
            "structured_data": None,
            "image_search_queries":None,
            "retrieved_image_data":None,
            "draft": None,
            "critique": None,
            "generated_content": None,
            "visual_generation_prompts": [],
            "visual_critique": None,
            "generated_audio_uri": None,
        }
        logger.info(f"Initial state prepared for game PK {latest_game_pk}.")


        # 4. --- Execute Agent via Streaming ---
        start_time = time.time()
        with stream_output_container: # Write stream into the container
             # Use st.write_stream with the generator function
             st.write_stream(run_agent_and_stream_progress(app, initial_state))
        end_time = time.time()
        logger.info(f"Agent streaming process finished UI-side in {end_time - start_time:.2f} seconds.")
        # NOTE: The helper function run_agent_and_stream_progress now handles
        # setting st.session_state['final_state'] and st.session_state['error_message']


    except Exception as e:
        # Catch errors happening *outside* the stream (like game_pk lookup or state setup)
        logger.error(f"An unexpected error occurred *before* agent streaming started for team {selected_team_display_name}: {e}", exc_info=True)
        st.session_state.error_message = f"❌ An unexpected error occurred before agent start: {e}"

    finally:
         # Mark run as complete regardless of success/error to trigger results display/error message
         st.session_state.run_complete = True
         st.rerun() # Rerun to display final results or the final error message


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
            st.warning("No audio URI found in the final state.")

        # --- Display Visual Assets (Images) ---
        st.subheader("🖼️ Visual Assets (Images)")
        all_image_assets = final_agent_state.get("all_image_assets") # Read from final_agent_state
        if all_image_assets and isinstance(all_image_assets, list):
            if not all_image_assets:
                 st.info("No image assets were found in the final state.")
            else:
                 cols = st.columns(3)
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
                                      st.image(signed_image_url, width=200)
                                      st.caption(f"Image loaded via temporary URL (expires).")
                                  except Exception as img_err:
                                      st.warning(f"Could not display image for {img_uri}. Error: {img_err}")
                                      st.markdown(f"Direct Link (expires): [View Image]({signed_image_url})")
                              else:
                                   st.warning(f"Could not generate temporary URL for image: {img_uri}")
                         elif img_uri:
                              st.warning(f"Image URI is not a GCS URI, attempting direct display: {img_uri}")
                              try: st.image(img_uri, width=200)
                              except: st.error(f"Failed to display non-GCS image URI.")
                         else:
                              st.warning("Image URI missing.")
                         st.divider()
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
                cols_vid = st.columns(2)
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
                                     st.video(signed_video_url)
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
                         st.divider()
                     col_idx_vid += 1
        else:
            st.warning("Video assets data is missing or not a list in the final state.")

        # --- Optionally display raw final state for debugging ---
        with st.expander("Show Raw Final State (for Debugging)"):
            debug_state = {k: v for k, v in final_agent_state.items() if k not in [
                'structured_data', 'narrative_context', 'draft',
                'generated_content', 'all_image_assets', 'all_video_assets',
                'generated_visual_assets', 'generated_video_assets',
                'player_lookup_dict']}
            try:
                st.json(json.dumps(debug_state, indent=2, default=str))
            except Exception:
                st.text(str(debug_state))

    # Display message if the run completed but no final state was captured (e.g., early error)
    elif not st.session_state.get("error_message"):
        st.warning("Agent run completed, but no final state was captured. Check logs for details.")

# Message shown before any run is attempted
elif not st.session_state.run_complete and not st.session_state.error_message:
    st.info("Select a team and click the button above to generate the game recap.")

# --- Footer or additional info ---
st.markdown("---")
st.markdown("Powered by LangGraph, Vertex AI (Gemini, Imagen, Veo, TTS), BigQuery, and GCS.")