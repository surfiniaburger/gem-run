# headshot_downloader_name_only.py

import requests
import os
import time
import logging
import re # <-- Import regex module for sanitization
from google.cloud import storage
from google.api_core.exceptions import NotFound
from ratelimit import limits, sleep_and_retry
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
GCP_PROJECT_ID = "silver-455021" #
GCS_BUCKET_NAME = "mlb-headshots-name-only" # <--- CONSIDER A NEW BUCKET/FOLDER to avoid mixing formats
GCS_FOLDER = "headshots"
MLB_API_SEASON = 2025 # Season to get rosters for
HEADSHOT_URL_TEMPLATE = "https://midfield.mlbstatic.com/v1/people/{person_id}/spots/120"
ROSTER_URL_TEMPLATE = "https://statsapi.mlb.com/api/v1/teams/{team_id}/roster?season={season}"

# Rate Limiting
MLB_API_CALLS = 9
MLB_API_RATE_LIMIT = 60
DOWNLOAD_CALLS = 30
DOWNLOAD_RATE_LIMIT = 60
MAX_WORKERS = 10

TEAMS = { # Your TEAMS dictionary
    'rangers': 140, 'angels': 108, 'astros': 117, 'rays': 139, 'blue_jays': 141,
    'yankees': 147, 'orioles': 110, 'red_sox': 111, 'twins': 142, 'white_sox': 145,
    'guardians': 114, 'tigers': 116, 'royals': 118, 'padres': 135, 'giants': 137,
    'diamondbacks': 109, 'rockies': 115, 'phillies': 143, 'braves': 144, 'marlins': 146,
    'nationals': 120, 'mets': 121, 'pirates': 134, 'cardinals': 138, 'brewers': 158,
    'cubs': 112, 'reds': 113, 'athletics': 133, 'mariners': 136, 'dodgers': 119,
}

# --- Initialize GCS Client ---
try:
    storage_client = storage.Client(project=GCP_PROJECT_ID)
    bucket = storage_client.bucket(GCS_BUCKET_NAME)
    logger.info(f"Initialized GCS Client for project '{GCP_PROJECT_ID}' and bucket '{GCS_BUCKET_NAME}'")
except Exception as e:
    logger.critical(f"Failed to initialize GCS Client: {e}", exc_info=True)
    exit(1)

# --- API Call Functions ---
@sleep_and_retry
@limits(calls=MLB_API_CALLS, period=MLB_API_RATE_LIMIT)
def call_mlb_roster_api(team_id: int, season: int) -> dict:
    # ... (implementation remains the same)
    url = ROSTER_URL_TEMPLATE.format(team_id=team_id, season=season)
    try:
        response = requests.get(url, timeout=20)
        response.raise_for_status()
        logger.debug(f"Successfully fetched roster for team {team_id}")
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching roster for team {team_id} from {url}: {e}")
        return {}

@sleep_and_retry
@limits(calls=DOWNLOAD_CALLS, period=DOWNLOAD_RATE_LIMIT)
def download_headshot(person_id: int) -> bytes | None:
    # ... (implementation remains the same)
    url = HEADSHOT_URL_TEMPLATE.format(person_id=person_id)
    try:
        response = requests.get(url, timeout=15)
        if response.status_code == 200:
            content_type = response.headers.get('Content-Type', '').lower()
            if 'image' in content_type:
                logger.debug(f"Successfully downloaded headshot for player {person_id}")
                return response.content
            else:
                logger.warning(f"Downloaded content for player {person_id} is not an image (Content-Type: {content_type}). Skipping.")
                return None
        elif response.status_code == 404:
            logger.warning(f"Headshot not found (404) for player {person_id} at {url}")
            return None
        else:
            logger.error(f"Error downloading headshot for player {person_id}. Status: {response.status_code}, URL: {url}")
            return None
    except requests.exceptions.Timeout:
         logger.error(f"Timeout downloading headshot for player {person_id} from {url}")
         return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading headshot for player {person_id} from {url}: {e}")
        return None


def upload_to_gcs(image_bytes: bytes, blob_name: str, content_type: str = 'image/jpeg'):
    # ... (implementation remains the same)
    try:
        blob = bucket.blob(blob_name)
        blob.upload_from_string(image_bytes, content_type=content_type)
        logger.info(f"Successfully uploaded {blob_name} to gs://{GCS_BUCKET_NAME}/{blob_name}")
        return True
    except Exception as e:
        logger.error(f"Failed to upload {blob_name} to GCS: {e}", exc_info=True)
        return False

def check_gcs_blob_exists(blob_name: str) -> bool:
    # ... (implementation remains the same)
    try:
        blob = bucket.blob(blob_name)
        return blob.exists()
    except Exception as e:
        logger.error(f"Error checking existence for blob {blob_name}: {e}", exc_info=True)
        return False

def sanitize_filename(name: str) -> str:
    """
    Converts a name into a safe ASCII-only filename string
    (lowercase, underscore separated).
    Removes non-ASCII characters.
    """
    if not name or name == 'Unknown':
        return ""

    # 1. Keep only ASCII characters
    try:
        # More robust way to filter ASCII, handles potential encoding nuances better
        ascii_name = name.encode('ascii', 'ignore').decode('ascii')
    except Exception as e:
        logger.warning(f"Could not filter ASCII for name '{name}': {e}. Skipping.")
        return "" # Return empty if filtering fails

    if not ascii_name:
         logger.warning(f"Name '{name}' became empty after ASCII filtering.")
         # Optionally return original name's sanitization attempt or skip
         # Let's try basic sanitization on original as fallback, but log heavily
         s_orig = name.lower()
         s_orig = re.sub(r'[^\w\s]+', '', s_orig) # Remove non-word/non-space
         s_orig = re.sub(r'\s+', '_', s_orig)    # Spaces to underscores
         s_orig = s_orig.strip('_')
         if not s_orig: return "" # Give up if still empty
         logger.warning(f"Falling back to basic sanitized original name: '{s_orig}'")
         s = s_orig # Use the fallback
    else:
        s = ascii_name.lower()

    # 2. Replace non-alphanumeric (excluding _) with underscore
    # \w matches letters, numbers, and underscore. This preserves underscores.
    s = re.sub(r'[^\w]+', '_', s)

    # 3. Replace multiple consecutive underscores with a single one
    s = re.sub(r'_+', '_', s)

    # 4. Remove leading/trailing underscores
    s = s.strip('_')

    if not s:
        # Log the original name if sanitization results in empty string
        logger.warning(f"Original name '{name}' resulted in empty string after full sanitization.")
        return ""
    return s


def process_player(player_info: dict):
    """Processes a single player: checks existence, downloads, uploads with NAME ONLY filename."""
    try:
        person_id = player_info.get('person', {}).get('id')
        full_name = player_info.get('person', {}).get('fullName', 'Unknown')

        if not person_id:
            logger.warning("Player info missing person ID. Skipping.")
            return 0

        # --- Generate Filename in headshot_NAME.jpg format ---
        sanitized_name = sanitize_filename(full_name)

        if sanitized_name:
            # Use the sanitized name ONLY
            file_name = f"headshot_{sanitized_name}.jpg" # <-- THE ONLY CHANGE NEEDED HERE
        else:
            # Cannot create a meaningful name-only filename if name is unknown
            logger.error(f"Player name '{full_name}' (ID: {person_id}) is invalid or unknown. Cannot generate name-only filename. Skipping player.")
            # Alternatively, fallback to ID: file_name = f"headshot_{person_id}.jpg", but this violates the user's goal for the embedding script.
            return 0 # Skip if name is unusable
        # -------------------------

        blob_name = f"{GCS_FOLDER}/{file_name}" if GCS_FOLDER else file_name

        # 1. Check if blob already exists in GCS
        if check_gcs_blob_exists(blob_name):
            logger.debug(f"Headshot gs://{GCS_BUCKET_NAME}/{blob_name} already exists. Skipping download/upload.")
            return 0

        # 2. Download headshot (rate limited)
        logger.info(f"Attempting download for {full_name} ({person_id}) -> {blob_name}")
        image_bytes = download_headshot(person_id)
        if not image_bytes:
            return 0

        # 3. Upload to GCS
        if upload_to_gcs(image_bytes, blob_name):
            return 1
        else:
            return 0

    except Exception as e:
        p_id = player_info.get('person', {}).get('id', 'N/A')
        p_name = player_info.get('person', {}).get('fullName', 'N/A')
        logger.error(f"Unexpected error processing player ID {p_id} (Name: {p_name}): {e}", exc_info=True)
        return 0

# --- Main Execution ---
if __name__ == "__main__":
    logger.info("--- Starting MLB Headshot Download and Upload Script (Filename: name_only) ---")
    # ... (rest of main execution remains the same as previous downloader script)
    total_players_processed = 0
    total_headshots_uploaded = 0
    start_time = time.time()

    all_players_to_process = []

    for team_name, team_id in TEAMS.items():
        logger.info(f"Fetching roster for {team_name.replace('_', ' ').title()} (ID: {team_id})...")
        roster_data = call_mlb_roster_api(team_id, MLB_API_SEASON)
        if roster_data and 'roster' in roster_data:
            players = roster_data['roster']
            logger.info(f"Found {len(players)} players on {team_name.replace('_', ' ').title()} roster.")
            all_players_to_process.extend(players)
        else:
            logger.warning(f"Could not retrieve or parse roster for team {team_id}.")

    logger.info(f"\nCollected a total of {len(all_players_to_process)} player entries across all teams.")

    unique_players_dict = {}
    for p in all_players_to_process:
        p_id = p.get('person', {}).get('id')
        if p_id and p_id not in unique_players_dict:
             unique_players_dict[p_id] = p

    logger.info(f"Processing {len(unique_players_dict)} unique players.\n")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_player = {executor.submit(process_player, player_info): player_info.get('person', {}).get('id')
                           for player_id, player_info in unique_players_dict.items()}
        processed_count = 0
        for future in as_completed(future_to_player):
            player_id = future_to_player[future]
            processed_count += 1
            try:
                uploads_made = future.result()
                total_headshots_uploaded += uploads_made
            except Exception as exc:
                p_name_for_error = unique_players_dict.get(player_id, {}).get('person', {}).get('fullName', 'N/A')
                logger.error(f'Player ID {player_id} (Name: {p_name_for_error}) generated an exception during processing: {exc}', exc_info=True)

            if processed_count % 50 == 0:
                 logger.info(f"Progress: Processed {processed_count}/{len(unique_players_dict)} players...")

    total_players_processed = processed_count
    end_time = time.time()
    duration = end_time - start_time
    logger.info("--- MLB Headshot Download and Upload Complete ---")
    logger.info(f"Processed {total_players_processed} unique players.")
    logger.info(f"Successfully uploaded {total_headshots_uploaded} new headshots.")
    logger.info(f"Total execution time: {duration:.2f} seconds")