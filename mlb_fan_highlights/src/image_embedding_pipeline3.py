# image_embedding_pipeline_name_only.py

import logging
import time
import re
import json
from typing import List, Dict, Any, Optional, Tuple, Set
import pandas as pd
from datetime import datetime, timezone
from ratelimit import limits, sleep_and_retry

from google.cloud import bigquery
from google.cloud import storage
from google.api_core.exceptions import NotFound, BadRequest, Conflict, PreconditionFailed
from google.api_core.retry import Retry

# --- Vertex AI Imports ---
import vertexai
from vertexai.vision_models import Image, MultiModalEmbeddingModel
# --- End Vertex AI Imports ---

# --- Placeholder for Roster Fetching (IF NEEDED - not strictly needed for embedding name-only files) ---
# If you still want the metadata table for other reasons, keep this section.
# Otherwise, you could remove roster fetching and player metadata table logic.
ROSTER_URL_TEMPLATE = "https://statsapi.mlb.com/api/v1/teams/{team_id}/roster?season={season}"
MLB_API_SEASON = 2024 # Or adjust as needed
TEAMS = { # Your TEAMS dictionary
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
GCP_LOCATION = "us-central1"
BQ_DATASET_LOCATION = "US"
BQ_DATASET_ID = "mlb_rag_data_2024"

# GCS Configuration
GCS_BUCKET_LOGOS = "mlb_logos" # Assumes logos remain the same
GCS_PREFIX_LOGOS = ""
GCS_BUCKET_HEADSHOTS = "mlb-headshots-name-only" # <--- POINT TO THE NEW BUCKET/FOLDER
GCS_PREFIX_HEADSHOTS = "headshots/"

# BigQuery Table/Index Names
EMBEDDING_TABLE_ID = "mlb_image_embeddings_sdk_name_only" # <--- New BQ Table
PLAYER_METADATA_TABLE_ID = "mlb_player_metadata_name_only" # <--- (Optional) New Metadata Table
VECTOR_INDEX_ID = "mlb_image_embeddings_sdk_name_only_idx" # <--- New Index

# Vertex AI Model Configuration
VERTEX_MULTIMODAL_MODEL_NAME = "multimodalembedding@001"
EMBEDDING_DIMENSIONALITY = 1408

# SDK Call Settings
SDK_RETRY_CONFIG = Retry(initial=1.0, maximum=10.0, multiplier=2.0, deadline=60.0)

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
def execute_bq_query(sql: str, job_config: Optional[bigquery.QueryJobConfig] = None) -> Optional[bigquery.table.RowIterator]:
    # ... (implementation remains the same)
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


# --- Setup Functions ---

def setup_embedding_table_sdk():
    """Creates the target table for storing SDK-generated embeddings (name-only format)."""
    full_table_id = f"{GCP_PROJECT_ID}.{BQ_DATASET_ID}.{EMBEDDING_TABLE_ID}"
    logger.info(f"Ensuring Embedding Table {full_table_id} exists for SDK results...")

    # Consider deleting existing table if schema changes or for clean tests
    try:
        logger.warning(f"Attempting to delete existing table {full_table_id}...")
        bq_client.delete_table(full_table_id, not_found_ok=True)
        logger.info(f"Table {full_table_id} deleted or did not exist.")
    except Exception as e:
        logger.error(f"Error attempting to delete table {full_table_id} (continuing to create): {e}", exc_info=True)

    schema = [
        bigquery.SchemaField("image_uri", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("image_type", "STRING", mode="NULLABLE"), # 'logo' or 'headshot'
        # Entity ID and Name will store the sanitized name for BOTH logos and headshots
        bigquery.SchemaField("entity_id", "STRING", mode="NULLABLE", description="Sanitized Team Name (logo) or Player Name (headshot)"),
        bigquery.SchemaField("entity_name", "STRING", mode="NULLABLE", description="Sanitized Team Name (logo) or Player Name (headshot)"),
        bigquery.SchemaField("embedding", "FLOAT64", mode="REPEATED"),
        bigquery.SchemaField("last_updated", "TIMESTAMP", mode="NULLABLE"),
    ]
    table = bigquery.Table(full_table_id, schema=schema)
    table.description = "Stores multimodal embeddings (SDK generated) using sanitized names as IDs for logos and headshots"

    try:
        bq_client.create_table(table)
        logger.info(f"Table {full_table_id} created with the latest schema.")
    except Conflict:
         logger.info(f"Table {full_table_id} already exists.")
    except Exception as e:
        logger.error(f"Failed to create embedding table {full_table_id}: {e}", exc_info=True)
        raise

# --- Player Metadata Functions (OPTIONAL based on need) ---
# If you no longer need the player_id -> player_name mapping in BQ for this pipeline,
# you can comment out or remove setup_player_metadata_table and populate_player_metadata.
# Keep them if they serve other purposes.

def setup_player_metadata_table():
    """(Optional) Creates the table to store player ID and name mappings."""
    # ... (Implementation remains the same, using PLAYER_METADATA_TABLE_ID) ...
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
    """(Optional) Fetches all rosters and upserts player ID/Name into the metadata table."""
    # ... (Implementation remains the same, using PLAYER_METADATA_TABLE_ID) ...
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
        time.sleep(0.2) # Be nice to API

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
    # ... (implementation remains the same)
    uris = set()
    try:
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


# In image_embedding_pipeline_name_only.py

def generate_embeddings_sdk(gcs_bucket_name: str, gcs_prefix: str, image_type: str, existing_uris: Set[str]) -> List[Dict]:
    """
    Lists images in GCS, generates embeddings using Vertex AI SDK, parses metadata
    using sanitized NAME as entity_id/entity_name for both logos and headshots.
    Uses robust regex for parsing sanitized names.
    """
    results_list = []
    bucket = storage_client.bucket(gcs_bucket_name)
    if gcs_prefix and not gcs_prefix.endswith('/'):
        gcs_prefix += '/'
    blobs = storage_client.list_blobs(bucket, prefix=gcs_prefix)
    count = 0
    processed_count = 0
    error_count = 0
    max_errors_to_log = 10

    logger.info(f"Processing images from gs://{gcs_bucket_name}/{gcs_prefix}...")

    for blob in blobs:
        count += 1
        image_uri = f"gs://{gcs_bucket_name}/{blob.name}"

        # --- Skips remain the same ---
        if image_uri in existing_uris: continue
        if blob.name.endswith('/'): continue
        if not blob.content_type or not blob.content_type.startswith("image/"):
             logger.warning(f"Skipping non-image file: {image_uri} (Content-Type: {blob.content_type})")
             continue

        relative_path = blob.name[len(gcs_prefix):] if gcs_prefix else blob.name
        logger.info(f"Processing image {processed_count + 1}: {image_uri} (Relative: {relative_path})")

        try:
            # --- Embedding generation remains the same ---
            vertex_image = Image.load_from_file(image_uri)
            response = multimodal_embedding_model.get_embeddings(
                image=vertex_image, contextual_text=None, dimension=EMBEDDING_DIMENSIONALITY
            )
            embedding_list = response.image_embedding

            # --- Parse Metadata using NAME only ---
            parsed_entity_id_name = None

            if image_type == 'logo':
                name_match = re.search(r'mlb-([a-z0-9-]+(?:-[a-z0-9-]+)*)-logo\.(png|jpg|jpeg)$', relative_path, re.IGNORECASE)
                if name_match:
                    parsed_entity_id_name = name_match.group(1)
                    logger.info(f" -> Parsed logo name: {parsed_entity_id_name}")
                else:
                    logger.warning(f"Could not parse logo name from relative path: {relative_path}")

            elif image_type == 'headshot':
                 # MODIFIED REGEX: Use \w+ which matches letters (including Unicode in Py3), numbers, and underscore.
                 # This robustly handles the output of the improved sanitize_filename.
                 name_match = re.search(r'headshot_(\w+)\.(jpg|jpeg|png)$', relative_path, re.IGNORECASE)
                 if name_match:
                     parsed_entity_id_name = name_match.group(1) # e.g., "andres_gimenez"
                     logger.info(f" -> Parsed headshot name: {parsed_entity_id_name}")
                 else:
                     # Keep the fallback check for older ID-only files if you might have them mixed in
                     id_only_match = re.search(r'headshot_(\d+)\.(jpg|jpeg|png)$', relative_path, re.IGNORECASE)
                     if id_only_match:
                         logger.warning(f"Found ID-only headshot file: {relative_path}. Cannot determine player name from filename. Skipping.")
                         parsed_entity_id_name = None
                     else:
                         logger.warning(f"Could not parse headshot name from relative path using expected patterns: {relative_path}")

            # --- End Metadata Parsing ---

            # --- Appending results remains the same ---
            if parsed_entity_id_name:
                results_list.append({
                    "image_uri": image_uri, "image_type": image_type,
                    "entity_id": parsed_entity_id_name, "entity_name": parsed_entity_id_name,
                    "embedding": embedding_list, "last_updated": datetime.now(timezone.utc)
                })
                processed_count += 1
            else:
                 logger.warning(f"Skipping embedding for {image_uri} due to missing entity identifier from filename.")

        # --- Error handling remains the same ---
        except PreconditionFailed as e:
             error_count += 1
             logger.error(f"PreconditionFailed error processing {image_uri}: {e}", exc_info=True if error_count <= max_errors_to_log else False)
        except Exception as e:
            error_count += 1
            logger.error(f"Error processing {image_uri}: {e}", exc_info=True if error_count <= max_errors_to_log else False)

        if processed_count > 0 and processed_count % 100 == 0:
             logger.info(f"Progress: Processed {processed_count} new {image_type} images...")


    logger.info(f"Finished processing gs://{gcs_bucket_name}/{gcs_prefix}. Found {count} blobs, processed {processed_count} new images successfully, encountered {error_count} errors.")
    return results_list

def load_embeddings_to_bq(results: List[Dict]):
    # ... (implementation remains the same, uses EMBEDDING_TABLE_ID) ...
    if not results:
        logger.info("No new embeddings generated to load.")
        return

    full_table_id = f"{GCP_PROJECT_ID}.{BQ_DATASET_ID}.{EMBEDDING_TABLE_ID}"
    logger.info(f"Preparing to load {len(results)} new embeddings into {full_table_id}...")

    df = pd.DataFrame(results)
    df['last_updated'] = pd.to_datetime(df['last_updated'])

    job_config = bigquery.LoadJobConfig(
        schema=[ # Ensure this matches setup_embedding_table_sdk schema
            bigquery.SchemaField("image_uri", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("image_type", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("entity_id", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("entity_name", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("embedding", "FLOAT64", mode="REPEATED"),
            bigquery.SchemaField("last_updated", "TIMESTAMP", mode="NULLABLE"),
        ],
        write_disposition="WRITE_APPEND",
    )

    try:
        load_job = bq_client.load_table_from_dataframe(df, full_table_id, job_config=job_config)
        logger.info(f"Starting BigQuery load job {load_job.job_id}...")
        load_job.result()
        logger.info(f"Successfully loaded {load_job.output_rows} rows into {full_table_id}.")
    except Exception as e:
        logger.error(f"Failed to load embeddings DataFrame to BigQuery: {e}", exc_info=True)


# --- Vector Index and Search Functions ---

def setup_vector_index_sdk():
    # ... (implementation remains the same, uses EMBEDDING_TABLE_ID and VECTOR_INDEX_ID) ...
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

# In image_embedding_pipeline_name_only.py

import json
# ... other imports
# In image_embedding_pipeline_name_only.py

import json # Ensure json is imported
# ... other imports

def search_similar_images_sdk(
    query_text: str,
    top_k: int = 1,
    filter_image_type: Optional[str] = None,
    filter_entity_id: Optional[str] = None,
    filter_entity_name: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Performs vector search on the SDK embedding table using a text query.
    Allows filtering by image_type, entity_id, and/or entity_name before vector search.
    Correctly formats the options parameter as a SQL string literal containing JSON,
    with inner single quotes escaped (' -> '').
    """
    logger.info(
        f"Performing vector search on SDK table for: '{query_text}', top_k={top_k}, "
        f"filter_type='{filter_image_type}', filter_id='{filter_entity_id}', filter_name='{filter_entity_name}'"
    )
    embedding_table_ref = f"`{GCP_PROJECT_ID}.{BQ_DATASET_ID}.{EMBEDDING_TABLE_ID}`"
    results = []
    vector_search_sql = "" # Initialize

    try:
        # 1. Generate query embedding (remains the same)
        logger.info("Generating embedding for search query...")
        # ... (embedding generation code remains the same) ...
        query_embedding = None
        try:
            query_response = multimodal_embedding_model.get_embeddings(
                contextual_text=query_text,
                dimension=EMBEDDING_DIMENSIONALITY,
            )
            query_embedding = query_response.text_embedding
            if not query_embedding: raise ValueError("SDK returned no text embedding.")
            logger.info("Query embedding generated via SDK.")
        except Exception as query_emb_err:
            logger.error(f"Failed to get query embedding via SDK: {query_emb_err}")
            return []

        if not query_embedding:
            logger.error("Failed to obtain query embedding.")
            return []

        query_embedding_str = f"[{','.join(map(str, query_embedding))}]"

        # 2. Build the raw SQL filter condition string (using single quotes)
        filter_conditions = []
        if filter_image_type:
            # Escape single quotes WITHIN the value (' -> '')
            safe_filter_value = filter_image_type.replace("'", "''")
            filter_conditions.append(f"image_type = '{safe_filter_value}'")
        if filter_entity_id:
            safe_filter_value = filter_entity_id.replace("'", "''")
            filter_conditions.append(f"entity_id = '{safe_filter_value}'")
        if filter_entity_name:
            safe_filter_value = filter_entity_name.replace("'", "''")
            filter_conditions.append(f"entity_name = '{safe_filter_value}'")

        # 3. Construct Options SQL Part
        options_sql_part = ""
        if filter_conditions:
            combined_filter_string = " AND ".join(filter_conditions)
            logger.info(f"Applying filter condition: {combined_filter_string}")
            # Example: entity_id = 'aaron_nola'
            # Example: image_type = 'headshot' AND entity_id = 'bryce_harper'

            # Escape the single quotes *within the combined filter string* for SQL nesting
            # Replace ' -> ''
            escaped_filter_string = combined_filter_string.replace("'", "''")
            # Example: entity_id = ''aaron_nola''
            # Example: image_type = ''headshot'' AND entity_id = ''bryce_harper''

            # Manually construct the JSON string within the SQL literal single quotes
            # Use double quotes for JSON keys/strings, use the escaped filter string
            options_sql_literal = f"'{{\"filter\": \"{escaped_filter_string}\"}}'"
            # Example result: '{"filter": "entity_id = ''aaron_nola''"}'
            # Example result: '{"filter": "image_type = ''headshot'' AND entity_id = ''bryce_harper''"}'

            options_sql_part = f",\n                options => {options_sql_literal}"

        # 4. Construct the final query
        vector_search_sql = f"""
        SELECT base.image_uri, base.image_type, base.entity_id, base.entity_name, distance
        FROM VECTOR_SEARCH(
                TABLE {embedding_table_ref},
                'embedding',
                (SELECT {query_embedding_str} AS embedding),
                top_k => {top_k},
                distance_type => 'COSINE'{options_sql_part} -- Options added here
            );
        """
        # logger.debug(f"Executing Vector Search SQL:\n{vector_search_sql}")

        search_results = execute_bq_query(vector_search_sql)

        # ... rest of the function ...
        if search_results:
            results = [dict(row.items()) for row in search_results]
            logger.info(f"Vector search returned {len(results)} results.")
        else:
            logger.warning("Vector search returned no results.")

    except Exception as e:
        logger.error(f"Error during vector search for '{query_text}': {e}", exc_info=True)
        try:
            logger.error(f"Failed Vector Search SQL:\n{vector_search_sql}")
        except NameError:
             logger.error("Could not log the failed SQL (error occurred before construction).")
        return []
    return results
# --- Main Execution Flow ---
# (No changes needed in the main block's example calls)
# --- Main Execution Flow ---
if __name__ == "__main__":
    logger.info("--- Starting MLB Image Embedding Pipeline (Name Only IDs) ---")
    full_start_time = time.time()

    # --- Step 0: (Optional) Setup and Populate Player Metadata ---
    # Decide if you still need this table for other purposes.
    # try:
    #     logger.info("\n=== Step 0: Setting up Player Metadata Table ===")
    #     setup_player_metadata_table()
    #     populate_player_metadata()
    # except Exception as e:
    #     logger.critical(f"Failed during Player Metadata setup: {e}. Aborting.", exc_info=True)
    #     exit(1)

    # --- Step 1: Setup BQ Embedding Table ---
    # try:
    #     logger.info("\n=== Step 1: Setting up Target BigQuery Embedding Table ===")
    #     setup_start_time = time.time()
    #     setup_embedding_table_sdk() # Creates the new name-only table
    #     logger.info(f"Target Embedding table setup took {time.time() - setup_start_time:.2f} seconds.")
    # except Exception as e:
    #     logger.critical(f"Failed during BigQuery embedding table setup: {e}. Aborting.", exc_info=True)
    #     exit(1)

    # # --- Step 2: Generate Embeddings via SDK and Load to BQ ---
    # try:
    #     logger.info("\n=== Step 2: Generating Embeddings via SDK and Loading to BigQuery ===")
    #     embed_start_time = time.time()

    #     target_table_fqn = f"{GCP_PROJECT_ID}.{BQ_DATASET_ID}.{EMBEDDING_TABLE_ID}"
    #     existing_uris_set = get_existing_uris(target_table_fqn)

    #     # Process Logos (uses name as ID)
    #     logo_results = generate_embeddings_sdk(
    #         gcs_bucket_name=GCS_BUCKET_LOGOS,
    #         gcs_prefix=GCS_PREFIX_LOGOS,
    #         image_type='logo',
    #         existing_uris=existing_uris_set
    #     )

    #     # Process Headshots (uses name as ID)
    #     headshot_results = generate_embeddings_sdk(
    #         gcs_bucket_name=GCS_BUCKET_HEADSHOTS, # Point to the name-only bucket
    #         gcs_prefix=GCS_PREFIX_HEADSHOTS,
    #         image_type='headshot',
    #         existing_uris=existing_uris_set
    #     )

    #     all_results = logo_results + headshot_results
    #     load_embeddings_to_bq(all_results)

    #     logger.info(f"SDK Embedding generation and BQ load took {time.time() - embed_start_time:.2f} seconds.")

    # except Exception as e:
    #     logger.error(f"Failed during SDK embedding generation or BQ load: {e}.", exc_info=True)
    #     logger.warning("Proceeding to index creation despite potential embedding errors.")


    # # --- Step 3: Setup Vector Index ---
    # try:
    #     logger.info("\n=== Step 3: Setting up Vector Index (Name Only Table) ===")
    #     index_start_time = time.time()
    #     setup_vector_index_sdk() # Creates index on the new name-only table
    #     logger.info(f"Vector index setup initiated took {time.time() - index_start_time:.2f} seconds (building happens async).")
    # except Exception as e:
    #     logger.error(f"Failed during vector index setup: {e}", exc_info=True)

    # --- Step 4: Example Search ---
    try:
        logger.info("\n=== Step 4: Example Vector Search (Name Only Table) ===")
        search_start_time = time.time()
        logger.info("Waiting 15 seconds before example search (index building takes time)...")
        time.sleep(15)

        # --- Example 1: Logo Search (No change needed) ---
        search_query_logo = "Arizona Diamondbacks logo"
        logo_results = search_similar_images_sdk(search_query_logo, top_k=1)
        print(f"\nSearch Results for '{search_query_logo}':")
        print(json.dumps(logo_results, indent=2, default=str))


        # --- Example 3: Headshot Search Filtered by Specific Player Name (Entity ID) ---
        # Use the *sanitized* name as stored in entity_id
        player_entity_id_to_find = "aaron_nola"
        search_query_specific_player = "pitcher throwing" # Query for visual similarity
        specific_player_results = search_similar_images_sdk(
            query_text=search_query_specific_player,
            top_k=1, # Find the single best match for this specific player
            filter_entity_id=player_entity_id_to_find # <-- FILTERING HERE
        )
        print(f"\nSpecific Player Search Results for '{search_query_specific_player}' (entity_id={player_entity_id_to_find}):")
        # Expecting results only for aaron_nola (if his image exists and matches query)
        print(json.dumps(specific_player_results, indent=2, default=str))

        # --- Example 4: Search combining type and entity filter ---
        search_query_bryce = "Bryce Harper headshot"
        bryce_results = search_similar_images_sdk(
            query_text=search_query_bryce,
            top_k=2,
            filter_image_type='headshot',
            filter_entity_id='bryce_harper' # Use the sanitized name
        )
        print(f"\nCombined Filter Search Results for '{search_query_bryce}' (type=headshot, entity_id=bryce_harper):")
        print(json.dumps(bryce_results, indent=2, default=str))


        logger.info(f"Example search execution took {time.time() - search_start_time - 15:.2f} seconds (excluding wait time).")
    except Exception as e:
        logger.error(f"Failed during example search: {e}", exc_info=True)

    logger.info(f"\n--- MLB Image Embedding Pipeline (Name Only IDs) Finished ---")
    logger.info(f"Total script execution time: {time.time() - full_start_time:.2f} seconds")