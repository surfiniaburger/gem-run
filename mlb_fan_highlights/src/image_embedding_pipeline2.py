import logging
import time
import re
import json
from typing import List, Dict, Any, Optional, Tuple, Set
import pandas as pd
from datetime import datetime, timezone
from ratelimit import limits, sleep_and_retry

from google.cloud import bigquery
from google.cloud import storage # Added GCS client
from google.api_core.exceptions import NotFound, BadRequest, Conflict, PreconditionFailed
from google.api_core.retry import Retry # Added for SDK calls

# --- Vertex AI Imports ---
import vertexai
from vertexai.vision_models import Image, MultiModalEmbeddingModel # Corrected import location
# --- End Vertex AI Imports ---

# --- Import roster fetching function (or copy it here) ---
# Assuming you have the roster fetching logic available
# Example: from headshot_downloader import call_mlb_roster_api, TEAMS
# --- Placeholder for Roster Fetching ---
# (Copy or import relevant parts from your headshot_downloader.py)
ROSTER_URL_TEMPLATE = "https://statsapi.mlb.com/api/v1/teams/{team_id}/roster?season={season}"
MLB_API_SEASON = 2024 # Or adjust as needed
TEAMS = {
    'rangers': 140, 'angels': 108, 'astros': 117, 'rays': 139, 'blue_jays': 141,
    'yankees': 147, 'orioles': 110, 'red_sox': 111, 'twins': 142, 'white_sox': 145,
    'guardians': 114, 'tigers': 116, 'royals': 118, 'padres': 135, 'giants': 137,
    'diamondbacks': 109, 'rockies': 115, 'phillies': 143, 'braves': 144, 'marlins': 146,
    'nationals': 120, 'mets': 121, 'pirates': 134, 'cardinals': 138, 'brewers': 158,
    'cubs': 112, 'reds': 113, 'athletics': 133, 'mariners': 136, 'dodgers': 119,
}
MLB_API_CALLS = 9
MLB_API_RATE_LIMIT = 60
@sleep_and_retry
@limits(calls=MLB_API_CALLS, period=MLB_API_RATE_LIMIT)
def call_mlb_roster_api(team_id: int, season: int) -> dict:
    import requests
    url = ROSTER_URL_TEMPLATE.format(team_id=team_id, season=season)
    try:
        response = requests.get(url, timeout=20)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching roster for team {team_id} from {url}: {e}")
        return {}
# --- End Roster Fetching Placeholder ---


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
GCP_PROJECT_ID = "silver-455021"
GCP_LOCATION = "us-central1" # Location for Vertex AI initialization
BQ_DATASET_LOCATION = "US" # Actual location of your BQ dataset
BQ_DATASET_ID = "mlb_rag_data_2024"

# GCS Configuration
GCS_BUCKET_LOGOS = "mlb_logos"
GCS_PREFIX_LOGOS = "" # e.g., "logos/" if logos are in a subfolder
GCS_BUCKET_HEADSHOTS = "mlb-headshots"
GCS_PREFIX_HEADSHOTS = "headshots/" # e.g., "headshots/"

# BigQuery Table/Index Names
# OBJECT_TABLE_ID = "mlb_images_object_table" # Not needed for this approach
# EMBEDDING_MODEL_ID = "mlb_multimodal_embedding_model" # Not needed for this approach
EMBEDDING_TABLE_ID = "mlb_image_embeddings_sdk" # Use a new table name to avoid conflicts
PLAYER_METADATA_TABLE_ID = "mlb_player_metadata"
VECTOR_INDEX_ID = "mlb_image_embeddings_sdk_idx" # Index name matches new table

# Vertex AI Model Configuration
VERTEX_MULTIMODAL_MODEL_NAME = "multimodalembedding@001"
EMBEDDING_DIMENSIONALITY = 1408

# SDK Call Settings
SDK_RETRY_CONFIG = Retry(initial=1.0, maximum=10.0, multiplier=2.0, deadline=60.0)
SDK_BATCH_SIZE = 10 # How many images to process before uploading a batch to BQ

# --- Initialize Clients ---
try:
    bq_client = bigquery.Client(project=GCP_PROJECT_ID, location=BQ_DATASET_LOCATION)
    storage_client = storage.Client(project=GCP_PROJECT_ID)
    vertexai.init(project=GCP_PROJECT_ID, location=GCP_LOCATION)
    multimodal_embedding_model = MultiModalEmbeddingModel.from_pretrained(VERTEX_MULTIMODAL_MODEL_NAME)
    logger.info(f"Initialized BQ, GCS, and Vertex AI clients/models.")
except Exception as e:
    logger.critical(f"Failed to initialize clients: {e}", exc_info=True)
    exit(1)

# --- Helper Function to Execute BQ Queries ---
# (Keep the execute_bq_query function as defined previously for schema checks, index, search)
def execute_bq_query(sql: str, job_config: Optional[bigquery.QueryJobConfig] = None) -> Optional[bigquery.table.RowIterator]:
    """Executes a BigQuery query and handles common errors."""
    try:
        logger.info(f"Executing BQ Query: {sql[:300]}...")
        query_job = bq_client.query(sql, job_config=job_config)
        results = query_job.result()
        logger.info("Query executed successfully.")
        return results
    except NotFound as e:
        logger.warning(f"Resource not found during query execution: {e}. Query: {sql[:150]}...")
        return None
    except BadRequest as e:
        if "Not found: Table" in str(e) or "Not found: Model" in str(e) or "Not found: Connection" in str(e):
             logger.warning(f"Resource (Table/Model/Connection) not found (BadRequest): {e}. Query: {sql[:150]}...")
             return None
        else:
            logger.error(f"BigQuery BadRequest error executing query: {e}", exc_info=True)
            logger.error(f"Query: {sql}")
            if hasattr(e, 'errors'): logger.error(f"  Errors: {e.errors}")
            raise
    except Conflict as e:
        if "Already Exists" in str(e):
             logger.info(f"Resource likely already exists (Conflict): {sql[:150]}...")
             return None
        else:
             logger.error(f"BigQuery Conflict error: {e}. Query: {sql[:150]}...")
             raise
    except Exception as e:
        logger.error(f"Unexpected error executing BQ query: {e}", exc_info=True)
        logger.error(f"Query: {sql}")
        raise

# --- Setup Functions (Simplified for SDK approach) ---

def setup_embedding_table_sdk():
    """Creates the target table for storing SDK-generated embeddings."""
    # *** Uses the new table name ***
    full_table_id = f"{GCP_PROJECT_ID}.{BQ_DATASET_ID}.{EMBEDDING_TABLE_ID}"
    logger.info(f"Ensuring Embedding Table {full_table_id} exists for SDK results...")

    # Delete existing table first to ensure schema is correct (optional, but safer for dev)
    try:
        logger.warning(f"Attempting to delete existing table {full_table_id}...")
        bq_client.delete_table(full_table_id, not_found_ok=True)
        logger.info(f"Table {full_table_id} deleted or did not exist.")
    except Exception as e:
        logger.error(f"Error attempting to delete table {full_table_id} (continuing to create): {e}", exc_info=True)

    # Define schema
    schema = [
        bigquery.SchemaField("image_uri", "STRING", mode="REQUIRED", description="GCS URI of the image"),
        bigquery.SchemaField("image_type", "STRING", mode="NULLABLE", description="Type of image: 'logo' or 'headshot'"),
        bigquery.SchemaField("entity_id", "STRING", mode="NULLABLE", description="Team ID/Name (from logo name) or Player ID (from headshot name)"),
        bigquery.SchemaField("entity_name", "STRING", mode="NULLABLE", description="Player Name (from metadata table) or Team Name (parsed)"),
        bigquery.SchemaField("embedding", "FLOAT64", mode="REPEATED", description=f"Multimodal embedding vector ({EMBEDDING_DIMENSIONALITY} dimensions)"),
        bigquery.SchemaField("last_updated", "TIMESTAMP", mode="NULLABLE"),
    ]
    table = bigquery.Table(full_table_id, schema=schema)
    table.description = "Stores multimodal embeddings (SDK generated) for MLB logos and player headshots"

    try:
        bq_client.create_table(table)
        logger.info(f"Table {full_table_id} created with the latest schema.")
    except Conflict:
         logger.info(f"Table {full_table_id} already exists (delete might have failed?).")
    except Exception as e:
        logger.error(f"Failed to create embedding table {full_table_id}: {e}", exc_info=True)
        raise

# --- Player Metadata Functions (Keep as before) ---
def setup_player_metadata_table():
    """Creates the table to store player ID and name mappings."""
    # (Keep implementation from previous version)
    full_table_id = f"{GCP_PROJECT_ID}.{BQ_DATASET_ID}.{PLAYER_METADATA_TABLE_ID}"
    logger.info(f"Ensuring Player Metadata Table {full_table_id} exists...")
    schema = [
        bigquery.SchemaField("player_id", "INTEGER", mode="REQUIRED"),
        bigquery.SchemaField("player_name", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("last_updated", "TIMESTAMP", mode="NULLABLE"),
    ]
    table = bigquery.Table(full_table_id, schema=schema)
    table.description = "Stores mapping between MLB Player ID and Full Name"
    try:
        bq_client.create_table(table, exists_ok=True)
        logger.info(f"Ensured table {full_table_id} exists or was created.")
    except Exception as e:
        logger.error(f"Failed to create or ensure player metadata table {full_table_id}: {e}", exc_info=True)
        raise

def populate_player_metadata():
    """Fetches all rosters and upserts player ID/Name into the metadata table."""
    # (Keep implementation from previous version, using MERGE)
    logger.info("Fetching all team rosters to update player metadata...")
    all_players_data = []
    for team_name, team_id in TEAMS.items():
        logger.debug(f"Fetching roster for {team_name.replace('_', ' ').title()} (ID: {team_id})...")
        roster_data = call_mlb_roster_api(team_id, MLB_API_SEASON)
        if roster_data and 'roster' in roster_data:
            for player in roster_data['roster']:
                person = player.get('person', {})
                player_id = person.get('id')
                player_name = person.get('fullName')
                if player_id and player_name:
                    all_players_data.append({'player_id': player_id, 'player_name': player_name})
        else:
            logger.warning(f"Could not retrieve or parse roster for team {team_id}.")
        time.sleep(0.2)

    if not all_players_data:
        logger.error("No player data fetched. Cannot populate player metadata table.")
        return

    players_df = pd.DataFrame(all_players_data)
    players_df = players_df.drop_duplicates(subset=['player_id'])
    players_df['player_id'] = pd.to_numeric(players_df['player_id'], errors='coerce').astype('Int64')
    players_df = players_df.dropna(subset=['player_id'])
    players_df['last_updated'] = pd.Timestamp.now(tz=timezone.utc)

    if players_df.empty:
        logger.warning("Player DataFrame is empty after cleaning.")
        return

    logger.info(f"Found {len(players_df)} unique players to upsert into metadata table.")
    temp_table_id = f"{PLAYER_METADATA_TABLE_ID}_temp_{int(time.time())}"
    full_temp_table_id = f"{GCP_PROJECT_ID}.{BQ_DATASET_ID}.{temp_table_id}"
    target_table_id = f"{GCP_PROJECT_ID}.{BQ_DATASET_ID}.{PLAYER_METADATA_TABLE_ID}"
    try:
        job_config_load = bigquery.LoadJobConfig(
            schema=[
                 bigquery.SchemaField("player_id", "INTEGER"),
                 bigquery.SchemaField("player_name", "STRING"),
                 bigquery.SchemaField("last_updated", "TIMESTAMP"),
            ], write_disposition="WRITE_TRUNCATE",
        )
        logger.info(f"Loading player data to temporary table {full_temp_table_id}...")
        load_job = bq_client.load_table_from_dataframe(players_df, full_temp_table_id, job_config=job_config_load)
        load_job.result()
        logger.info(f"Loaded {load_job.output_rows} rows to temporary table.")

        merge_sql = f"""
        MERGE `{target_table_id}` T USING `{full_temp_table_id}` S ON T.player_id = S.player_id
        WHEN MATCHED THEN UPDATE SET T.player_name = S.player_name, T.last_updated = S.last_updated
        WHEN NOT MATCHED THEN INSERT (player_id, player_name, last_updated) VALUES(S.player_id, S.player_name, S.last_updated);
        """
        logger.info("Executing MERGE statement to update player metadata...")
        merge_job = bq_client.query(merge_sql)
        merge_job.result()
        logger.info("Player metadata table successfully updated.")
    except Exception as e:
        logger.error(f"Error during player metadata upsert: {e}", exc_info=True)
    finally:
        try:
            logger.info(f"Deleting temporary table {full_temp_table_id}...")
            bq_client.delete_table(full_temp_table_id, not_found_ok=True)
        except Exception as e:
            logger.error(f"Error deleting temporary table {full_temp_table_id}: {e}", exc_info=True)


# --- SDK Embedding Generation and Loading ---

def get_existing_uris(target_table_id: str) -> Set[str]:
    """Fetches all existing image URIs from the target embedding table."""
    uris = set()
    try:
        # Ensure the table exists before querying
        bq_client.get_table(target_table_id)
        query = f"SELECT DISTINCT image_uri FROM `{target_table_id}` WHERE image_uri IS NOT NULL"
        results = execute_bq_query(query)
        if results:
            for row in results:
                uris.add(row.image_uri)
        logger.info(f"Found {len(uris)} existing URIs in target table.")
    except NotFound:
        logger.info("Target embedding table does not exist yet, no existing URIs to fetch.")
    except Exception as e:
        logger.error(f"Error fetching existing URIs: {e}. Proceeding without exclusion.", exc_info=True)
    return uris

def generate_embeddings_sdk(gcs_bucket_name: str, gcs_prefix: str, image_type: str, player_lookup: Dict[int, str], existing_uris: Set[str]) -> List[Dict]:
    """
    Lists images in GCS, generates embeddings using Vertex AI SDK, parses metadata,
    and returns a list of dictionaries ready for DataFrame creation.
    """
    results_list = []
    bucket = storage_client.bucket(gcs_bucket_name)
    blobs = storage_client.list_blobs(bucket, prefix=gcs_prefix)
    count = 0
    processed_count = 0
    error_count = 0
    max_errors_to_log = 10

    logger.info(f"Processing images from gs://{gcs_bucket_name}/{gcs_prefix}...")

    for blob in blobs:
        count += 1
        image_uri = f"gs://{gcs_bucket_name}/{blob.name}"

        # Skip if already processed
        if image_uri in existing_uris:
            # logger.debug(f"Skipping already processed URI: {image_uri}")
            continue

        # Skip folders/non-image files if prefix is broad
        if blob.name.endswith('/'):
            continue
        if not blob.content_type or not blob.content_type.startswith("image/"):
             logger.warning(f"Skipping non-image file: {image_uri} (Content-Type: {blob.content_type})")
             continue

        logger.info(f"Processing image {processed_count + 1}: {image_uri}")

        try:
            # 1. Load Image
            # SDK's Image.load_from_file handles gs:// URIs
            vertex_image = Image.load_from_file(image_uri)

            # 2. Generate Embedding
            response = multimodal_embedding_model.get_embeddings(
                image=vertex_image,
                contextual_text=None, # No text context needed for basic image embedding
                dimension=EMBEDDING_DIMENSIONALITY,
            )
            embedding_list = response.image_embedding

            # 3. Parse Metadata
            parsed_entity_id_str = None
            entity_name = None
            if image_type == 'logo':
                 # Try ID first, then team name
                 id_match = re.search(r'/.*?(\d+).*\.(png|jpg|jpeg)$', blob.name, re.IGNORECASE)
                 name_match = re.search(r'/mlb-([a-z0-9-]+(?:-[a-z0-9-]+)*)-logo\.(png|jpg|jpeg)$', blob.name, re.IGNORECASE)
                 if id_match:
                     parsed_entity_id_str = id_match.group(1)
                 elif name_match:
                     parsed_entity_id_str = name_match.group(1)
                 entity_name = parsed_entity_id_str # For logos, use the ID/name as the name for now
            elif image_type == 'headshot':
                 id_match = re.search(r'/headshot_(\d+)\.(jpg|jpeg|png)$', blob.name, re.IGNORECASE)
                 if id_match:
                     parsed_entity_id_str = id_match.group(1)
                     try:
                         player_id_int = int(parsed_entity_id_str)
                         entity_name = player_lookup.get(player_id_int, None) # Lookup name
                         if not entity_name:
                              logger.warning(f"Player name not found in lookup for ID: {player_id_int}")
                     except ValueError:
                          logger.warning(f"Could not parse player ID as integer: {parsed_entity_id_str}")

            if not parsed_entity_id_str:
                 logger.warning(f"Could not parse entity ID for {image_uri}")

            # 4. Append result
            results_list.append({
                "image_uri": image_uri,
                "image_type": image_type,
                "entity_id": parsed_entity_id_str,
                "entity_name": entity_name,
                "embedding": embedding_list,
                "last_updated": datetime.now(timezone.utc)
            })
            processed_count += 1

            # Optional: Add a small sleep to avoid hitting potential API rate limits
            # time.sleep(0.1)

        except PreconditionFailed as e:
            # Often indicates image format issues or size limits for the model
             error_count += 1
             logger.error(f"PreconditionFailed error processing {image_uri}: {e}", exc_info=True if error_count <= max_errors_to_log else False)
             if error_count == max_errors_to_log: logger.error("Further SDK errors will not log full traceback.")
        except Exception as e:
            error_count += 1
            logger.error(f"Error processing {image_uri}: {e}", exc_info=True if error_count <= max_errors_to_log else False)
            if error_count == max_errors_to_log: logger.error("Further SDK errors will not log full traceback.")

        if processed_count > 0 and processed_count % 100 == 0:
             logger.info(f"Progress: Processed {processed_count} new {image_type} images...")


    logger.info(f"Finished processing {gcs_bucket_name}/{gcs_prefix}. Found {count} blobs, processed {processed_count} new images successfully, encountered {error_count} errors.")
    return results_list


def load_embeddings_to_bq(results: List[Dict]):
    """Loads the generated embedding data into BigQuery."""
    if not results:
        logger.info("No new embeddings generated to load.")
        return

    full_table_id = f"{GCP_PROJECT_ID}.{BQ_DATASET_ID}.{EMBEDDING_TABLE_ID}"
    logger.info(f"Preparing to load {len(results)} new embeddings into {full_table_id}...")

    df = pd.DataFrame(results)

    # Ensure correct data types for BQ schema
    df['last_updated'] = pd.to_datetime(df['last_updated'])
    # BQ client library handles list-like columns (embeddings) correctly if they are lists/tuples

    # Define BQ schema for load job (should match create_table schema)
    job_config = bigquery.LoadJobConfig(
        schema=[
            bigquery.SchemaField("image_uri", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("image_type", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("entity_id", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("entity_name", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("embedding", "FLOAT64", mode="REPEATED"),
            bigquery.SchemaField("last_updated", "TIMESTAMP", mode="NULLABLE"),
        ],
        write_disposition="WRITE_APPEND", # Append new embeddings
    )

    try:
        load_job = bq_client.load_table_from_dataframe(df, full_table_id, job_config=job_config)
        logger.info(f"Starting BigQuery load job {load_job.job_id}...")
        load_job.result() # Wait for completion
        logger.info(f"Successfully loaded {load_job.output_rows} rows into {full_table_id}.")
    except Exception as e:
        logger.error(f"Failed to load embeddings DataFrame to BigQuery: {e}", exc_info=True)
        # Consider saving the DataFrame locally if BQ load fails
        # df.to_csv("failed_embeddings_batch.csv", index=False)
        # logger.info("Saved failed batch to failed_embeddings_batch.csv")


# --- Vector Index and Search Functions (Keep as before, but use new table/index names) ---

def setup_vector_index_sdk():
    """Creates the Vector Index on the new SDK embedding table."""
    # *** Uses new table/index names ***
    embedding_table_fqn = f"`{GCP_PROJECT_ID}.{BQ_DATASET_ID}.{EMBEDDING_TABLE_ID}`"
    index_fqn = f"`{GCP_PROJECT_ID}.{BQ_DATASET_ID}`.{VECTOR_INDEX_ID}"
    index_check_sql = f"""
    SELECT index_name FROM `{GCP_PROJECT_ID}.{BQ_DATASET_ID}`.INFORMATION_SCHEMA.VECTOR_INDEXES
    WHERE table_name = '{EMBEDDING_TABLE_ID}' AND index_name = '{VECTOR_INDEX_ID}';
    """
    logger.info(f"Ensuring Vector Index {VECTOR_INDEX_ID} exists on {embedding_table_fqn}...")
    results = execute_bq_query(index_check_sql)
    if results and results.total_rows > 0:
         logger.info(f"Vector index {VECTOR_INDEX_ID} already exists.")
         return

    logger.info(f"Vector index {VECTOR_INDEX_ID} not found, creating...")
    create_index_sql = f"""
    CREATE OR REPLACE VECTOR INDEX {VECTOR_INDEX_ID}
    ON {embedding_table_fqn}(embedding)
    OPTIONS(distance_type='COSINE', index_type='IVF', ivf_options='{{"num_lists": 100}}');
    """
    execute_bq_query(create_index_sql)
    logger.info(f"Vector index {VECTOR_INDEX_ID} creation initiated (may take time to build).")


def search_similar_images_sdk(
    query_text: str, top_k: int = 5, filter_image_type: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Performs vector search on the SDK embedding table using a text query."""
    # *** Uses new table/index names ***
    logger.info(f"Performing vector search on SDK table for: '{query_text}', top_k={top_k}, filter='{filter_image_type}'")
    embedding_table_ref = f"`{GCP_PROJECT_ID}.{BQ_DATASET_ID}.{EMBEDDING_TABLE_ID}`"
    # Model is needed for the *query* embedding, use the SDK model instance
    results = []

    try:
        # 1. Generate query embedding using SDK
        logger.info("Generating embedding for search query using SDK...")
        # Note: The SDK might have different methods for text embedding if using MultiModal model
        # For simplicity, let's assume we still use it or a dedicated text model instance
        # query_embeddings = text_embedding_model.get_embeddings([query_text]) # If using text model
        # query_embedding = query_embeddings[0].values

        # Using the multimodal model for text query embedding (as in Colab)
        # This might require enabling specific APIs or checking model capabilities
        try:
             query_response = multimodal_embedding_model.get_embeddings(
                 contextual_text=query_text,
                 dimension=EMBEDDING_DIMENSIONALITY,
             )
             query_embedding = query_response.text_embedding
             if not query_embedding:
                  raise ValueError("Multimodal model did not return text embedding for query.")
        except Exception as query_emb_err:
             logger.error(f"Failed to get query embedding via SDK: {query_emb_err}. Trying BQ SQL fallback.")
             # Fallback to using BQ SQL if SDK fails for text query embedding
             try:
                logger.info("Attempting query embedding via BQ SQL fallback...")
                # Need the BQ remote model ID again for this fallback
                bq_model_ref = f"`{GCP_PROJECT_ID}.{BQ_DATASET_ID}.mlb_multimodal_embedding_model`" # Assume it exists or create it
                query_embedding_sql = f"""
                SELECT ml_generate_embedding_result
                FROM ML.GENERATE_EMBEDDING( MODEL {bq_model_ref}, (SELECT '{query_text}' AS content) )
                WHERE ml_generate_embedding_status = 'OK' LIMIT 1;
                """
                embedding_result = execute_bq_query(query_embedding_sql)
                if not embedding_result: raise ValueError("BQ SQL fallback failed.")
                query_embedding_list = list(embedding_result)
                if not query_embedding_list: raise ValueError("BQ SQL fallback returned no result.")
                query_embedding = query_embedding_list[0].ml_generate_embedding_result
             except Exception as fallback_err:
                  logger.error(f"BQ SQL fallback for query embedding also failed: {fallback_err}")
                  return [] # Cannot proceed without query embedding

        query_embedding_str = f"[{','.join(map(str, query_embedding))}]"
        logger.info("Query embedding generated successfully.")

        # 2. Perform Vector Search using BQ SQL
        filter_str = ""
        if filter_image_type:
            escaped_filter_type = filter_image_type.replace("'", "\\'")
            filter_str = f', options => \'{{"filter": "image_type=\'{escaped_filter_type}\'"}}\''

        vector_search_sql = f"""
        SELECT base.image_uri, base.image_type, base.entity_id, base.entity_name, distance
        FROM VECTOR_SEARCH(
                TABLE {embedding_table_ref}, 'embedding',
                (SELECT {query_embedding_str} AS embedding),
                top_k => {top_k}, distance_type => 'COSINE' {filter_str}
            );
        """
        search_results = execute_bq_query(vector_search_sql)

        if search_results:
            results = [dict(row.items()) for row in search_results]
            logger.info(f"Vector search returned {len(results)} results.")
        else:
            logger.warning("Vector search returned no results (check index status or query).")

    except Exception as e:
        logger.error(f"Error during vector search for '{query_text}': {e}", exc_info=True)
        return []

    return results


# --- Main Execution Flow ---
if __name__ == "__main__":
    logger.info("--- Starting MLB Image Embedding Pipeline (Vertex AI SDK Approach) ---")
    full_start_time = time.time()

    # --- Step 0: Setup and Populate Player Metadata ---
    try:
        logger.info("\n=== Step 0: Setting up Player Metadata Table ===")
        meta_start_time = time.time()
        setup_player_metadata_table()
        populate_player_metadata() # Run each time for freshness
        logger.info(f"Player metadata setup/update took {time.time() - meta_start_time:.2f} seconds.")
        # Load player metadata into a dictionary for faster lookups during embedding
        logger.info("Loading player metadata into memory for lookups...")
        player_lookup_query = f"SELECT player_id, player_name FROM `{GCP_PROJECT_ID}.{BQ_DATASET_ID}.{PLAYER_METADATA_TABLE_ID}`"
        player_results = execute_bq_query(player_lookup_query)
        player_lookup_dict = {row.player_id: row.player_name for row in player_results} if player_results else {}
        logger.info(f"Loaded {len(player_lookup_dict)} player names into lookup dictionary.")

    except Exception as e:
        logger.critical(f"Failed during Player Metadata setup: {e}. Aborting.", exc_info=True)
        exit(1)

    # --- Step 1: Setup BQ Embedding Table ---
    try:
        logger.info("\n=== Step 1: Setting up Target BigQuery Embedding Table ===")
        setup_start_time = time.time()
        setup_embedding_table_sdk() # Use the SDK version
        logger.info(f"Target Embedding table setup took {time.time() - setup_start_time:.2f} seconds.")
    except Exception as e:
        logger.critical(f"Failed during BigQuery embedding table setup: {e}. Aborting.", exc_info=True)
        exit(1)

    # --- Step 2: Generate Embeddings via SDK and Load to BQ ---
    try:
        logger.info("\n=== Step 2: Generating Embeddings via SDK and Loading to BigQuery ===")
        embed_start_time = time.time()

        # Get URIs already in the target table to avoid reprocessing
        target_table_fqn = f"{GCP_PROJECT_ID}.{BQ_DATASET_ID}.{EMBEDDING_TABLE_ID}"
        existing_uris_set = get_existing_uris(target_table_fqn)

        # Process Logos
        logo_results = generate_embeddings_sdk(
            gcs_bucket_name=GCS_BUCKET_LOGOS,
            gcs_prefix=GCS_PREFIX_LOGOS,
            image_type='logo',
            player_lookup=player_lookup_dict, # Not used for logos but pass anyway
            existing_uris=existing_uris_set
        )

        # Process Headshots
        headshot_results = generate_embeddings_sdk(
            gcs_bucket_name=GCS_BUCKET_HEADSHOTS,
            gcs_prefix=GCS_PREFIX_HEADSHOTS,
            image_type='headshot',
            player_lookup=player_lookup_dict,
            existing_uris=existing_uris_set
        )

        # Combine results
        all_results = logo_results + headshot_results

        # Load combined results to BigQuery
        load_embeddings_to_bq(all_results)

        logger.info(f"SDK Embedding generation and BQ load took {time.time() - embed_start_time:.2f} seconds.")

    except Exception as e:
        logger.error(f"Failed during SDK embedding generation or BQ load: {e}.", exc_info=True)
        logger.warning("Proceeding to index creation despite potential embedding errors.")


    # --- Step 3: Setup Vector Index (on the new table) ---
    try:
        logger.info("\n=== Step 3: Setting up Vector Index (SDK Table) ===")
        index_start_time = time.time()
        setup_vector_index_sdk() # Use the SDK version
        logger.info(f"Vector index setup initiated took {time.time() - index_start_time:.2f} seconds (building happens async).")
    except Exception as e:
        logger.error(f"Failed during vector index setup: {e}", exc_info=True)

    # --- Step 4: Example Search (on the new table) ---
    try:
        logger.info("\n=== Step 4: Example Vector Search (SDK Table) ===")
        search_start_time = time.time()
        logger.info("Waiting 90 seconds before example search (index building takes time)...")
        time.sleep(90)

        search_query_logo = "Arizona Diamondbacks logo"
        logo_results = search_similar_images_sdk(search_query_logo, top_k=3, filter_image_type='logo')
        print(f"\nSearch Results for '{search_query_logo}':")
        print(json.dumps(logo_results, indent=2, default=str))

        search_query_headshot = "player Mookie Betts"
        headshot_results = search_similar_images_sdk(search_query_headshot, top_k=3, filter_image_type='headshot')
        print(f"\nSearch Results for '{search_query_headshot}':")
        print(json.dumps(headshot_results, indent=2, default=str))

        logger.info(f"Example search execution took {time.time() - search_start_time:.2f} seconds (excluding wait time).")
    except Exception as e:
        logger.error(f"Failed during example search: {e}", exc_info=True)


    logger.info(f"\n--- MLB Image Embedding Pipeline (SDK Approach) Finished ---")
    logger.info(f"Total script execution time: {time.time() - full_start_time:.2f} seconds")