import requests
import os
import time
import logging
from google.cloud import storage
from google.api_core.exceptions import NotFound
from ratelimit import limits, sleep_and_retry
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
GCP_PROJECT_ID = "silver-455021" # 
GCS_BUCKET_NAME = "mlb-headshots" 
GCS_FOLDER = "headshots" # Optional: Subfolder within the bucket
MLB_API_SEASON = 2024 # Season to get rosters for
HEADSHOT_URL_TEMPLATE = "https://midfield.mlbstatic.com/v1/people/{person_id}/spots/120"
ROSTER_URL_TEMPLATE = "https://statsapi.mlb.com/api/v1/teams/{team_id}/roster?season={season}"

# Rate Limiting (adjust as needed, be respectful of the API)
MLB_API_CALLS = 9  # Max calls per minute to MLB Stats API (roster endpoint)
MLB_API_RATE_LIMIT = 60 # Seconds (1 minute)
DOWNLOAD_CALLS = 30 # Max headshot downloads per minute
DOWNLOAD_RATE_LIMIT = 60 # Seconds

MAX_WORKERS = 10 # Number of parallel threads for downloading/uploading

TEAMS = {
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

# --- API Call Functions with Rate Limiting ---
@sleep_and_retry
@limits(calls=MLB_API_CALLS, period=MLB_API_RATE_LIMIT)
def call_mlb_roster_api(team_id: int, season: int) -> dict:
    """Calls the MLB Roster API with rate limiting."""
    url = ROSTER_URL_TEMPLATE.format(team_id=team_id, season=season)
    try:
        response = requests.get(url, timeout=20)
        response.raise_for_status()
        logger.debug(f"Successfully fetched roster for team {team_id}")
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching roster for team {team_id} from {url}: {e}")
        return {} # Return empty dict on error

@sleep_and_retry
@limits(calls=DOWNLOAD_CALLS, period=DOWNLOAD_RATE_LIMIT)
def download_headshot(person_id: int) -> bytes | None:
    """Downloads headshot image bytes with rate limiting."""
    url = HEADSHOT_URL_TEMPLATE.format(person_id=person_id)
    try:
        response = requests.get(url, timeout=15)
        if response.status_code == 200:
            # Basic check for image content type
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
    """Uploads image bytes to GCS."""
    try:
        blob = bucket.blob(blob_name)
        blob.upload_from_string(image_bytes, content_type=content_type)
        logger.info(f"Successfully uploaded {blob_name} to gs://{GCS_BUCKET_NAME}/{blob_name}")
        return True
    except Exception as e:
        logger.error(f"Failed to upload {blob_name} to GCS: {e}", exc_info=True)
        return False

def check_gcs_blob_exists(blob_name: str) -> bool:
    """Checks if a blob exists in the GCS bucket."""
    try:
        blob = bucket.blob(blob_name)
        return blob.exists()
    except Exception as e:
        logger.error(f"Error checking existence for blob {blob_name}: {e}", exc_info=True)
        return False # Assume it doesn't exist if check fails

def process_player(player_info: dict):
    """Processes a single player: checks existence, downloads, uploads."""
    try:
        person_id = player_info.get('person', {}).get('id')
        full_name = player_info.get('person', {}).get('fullName', 'Unknown')
        status = player_info.get('status', {}).get('code', '').upper()

        if not person_id:
            logger.warning("Player info missing person ID. Skipping.")
            return 0

        # Only process active players ('A') - adjust if you need others (e.g., 'IL')
        # if status != 'A':
        #     logger.debug(f"Skipping non-active player {full_name} ({person_id}), status: {status}")
        #     return 0

        # Define GCS blob name
        file_name = f"headshot_{person_id}.jpg" # Assuming JPG, adjust if needed
        blob_name = f"{GCS_FOLDER}/{file_name}" if GCS_FOLDER else file_name

        # 1. Check if blob already exists in GCS
        if check_gcs_blob_exists(blob_name):
            logger.debug(f"Headshot gs://{GCS_BUCKET_NAME}/{blob_name} already exists. Skipping download/upload.")
            return 0 # 0 uploads needed

        # 2. Download headshot (rate limited)
        logger.info(f"Attempting download for {full_name} ({person_id})")
        image_bytes = download_headshot(person_id)
        if not image_bytes:
            return 0 # Failed download

        # 3. Upload to GCS
        if upload_to_gcs(image_bytes, blob_name):
            return 1 # 1 successful upload
        else:
            return 0 # Failed upload

    except Exception as e:
        logger.error(f"Unexpected error processing player ID {person_id}: {e}", exc_info=True)
        return 0

# --- Main Execution ---
if __name__ == "__main__":
    logger.info("--- Starting MLB Headshot Download and Upload Script ---")
    total_players_processed = 0
    total_headshots_uploaded = 0
    start_time = time.time()

    all_players_to_process = []

    # 1. Get all players from all teams first
    for team_name, team_id in TEAMS.items():
        logger.info(f"Fetching roster for {team_name.replace('_', ' ').title()} (ID: {team_id})...")
        roster_data = call_mlb_roster_api(team_id, MLB_API_SEASON)

        if roster_data and 'roster' in roster_data:
            players = roster_data['roster']
            logger.info(f"Found {len(players)} players on {team_name.replace('_', ' ').title()} roster.")
            all_players_to_process.extend(players)
        else:
            logger.warning(f"Could not retrieve or parse roster for team {team_id}.")
        # Minimal sleep between roster fetches if needed, rate limit decorator handles main delay
        # time.sleep(0.5)

    logger.info(f"\nCollected a total of {len(all_players_to_process)} player entries across all teams.")
    unique_player_ids = {p.get('person', {}).get('id') for p in all_players_to_process if p.get('person', {}).get('id')}
    logger.info(f"Processing {len(unique_player_ids)} unique players.\n")

    # Use a dictionary to ensure we only process each player once, even if on multiple rosters (rare)
    unique_players_dict = {p.get('person', {}).get('id'): p for p in all_players_to_process if p.get('person', {}).get('id')}


    # 2. Process players in parallel
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_player = {executor.submit(process_player, player_info): player_info.get('person', {}).get('id')
                           for player_id, player_info in unique_players_dict.items()}

        for future in as_completed(future_to_player):
            player_id = future_to_player[future]
            total_players_processed += 1
            try:
                uploads_made = future.result() # Will be 1 if uploaded, 0 otherwise
                total_headshots_uploaded += uploads_made
            except Exception as exc:
                logger.error(f'Player ID {player_id} generated an exception: {exc}')

            if total_players_processed % 50 == 0: # Log progress periodically
                 logger.info(f"Progress: Processed {total_players_processed}/{len(unique_players_dict)} players...")


    end_time = time.time()
    duration = end_time - start_time
    logger.info("--- MLB Headshot Download and Upload Complete ---")
    logger.info(f"Processed {total_players_processed} unique players.")
    logger.info(f"Successfully uploaded {total_headshots_uploaded} new headshots.")
    logger.info(f"Total execution time: {duration:.2f} seconds")