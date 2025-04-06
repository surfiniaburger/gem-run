import logging
import time
import re
import json
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd # Added for processing player data
from ratelimit import limits, sleep_and_retry

from google.cloud import bigquery
from google.api_core.exceptions import NotFound, BadRequest, Conflict

# --- Import roster fetching function (or copy it here) ---
# Assuming you have the roster fetching logic available, e.g., from your downloader script
# If not, you'll need to copy/adapt call_mlb_roster_api, TEAMS, etc. here.
# Example: from headshot_downloader import call_mlb_roster_api, TEAMS
# For simplicity, let's assume the necessary parts are included below for now.

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
# --- Add Rate Limiting if calling API here ---
# from ratelimit import limits, sleep_and_retry
MLB_API_CALLS = 9
MLB_API_RATE_LIMIT = 60
@sleep_and_retry
@limits(calls=MLB_API_CALLS, period=MLB_API_RATE_LIMIT)
def call_mlb_roster_api(team_id: int, season: int) -> dict:
    """Calls the MLB Roster API.""" # Add rate limiting if needed
    import requests # Add import if not already present
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
BQ_DATASET_LOCATION = "US" # Location of your BigQuery dataset
BQ_DATASET_ID = "mlb_rag_data_2024"

# GCS Configuration
GCS_LOGOS_URI = "gs://mlb_logos/*"
GCS_HEADSHOTS_URI = "gs://mlb-headshots/headshots/*"

# Connection Configuration (Using the single connection)
CONNECTION_LOCATION = "us" # Location where bq-vertex-ai-connector exists
CONNECTION_ID = "bq-vertex-ai-connector"
REUSED_CONNECTION_NAME = f"{GCP_PROJECT_ID}.{CONNECTION_LOCATION}.{CONNECTION_ID}"

# BigQuery Table/Model/Index Names
OBJECT_TABLE_ID = "mlb_images_object_table"
EMBEDDING_MODEL_ID = "mlb_multimodal_embedding_model"
EMBEDDING_TABLE_ID = "mlb_image_embeddings"
PLAYER_METADATA_TABLE_ID = "mlb_player_metadata" # <-- New Table
VECTOR_INDEX_ID = "mlb_image_embeddings_idx"

# Model Configuration
VERTEX_MULTIMODAL_ENDPOINT = "multimodalembedding@001"
EMBEDDING_DIMENSIONALITY = 1408

# --- Initialize BigQuery Client ---
try:
    bq_client = bigquery.Client(project=GCP_PROJECT_ID, location=BQ_DATASET_LOCATION)
    logger.info(f"Initialized BigQuery client for project {GCP_PROJECT_ID} (default location: {BQ_DATASET_LOCATION})")
except Exception as e:
    logger.critical(f"Failed to initialize BigQuery client: {e}", exc_info=True)
    exit(1)

# --- Helper Function to Execute BQ Queries ---
def execute_bq_query(sql: str, job_config: Optional[bigquery.QueryJobConfig] = None) -> Optional[bigquery.table.RowIterator]:
    """Executes a BigQuery query and handles common errors."""
    try:
        logger.info(f"Executing BQ Query: {sql[:300]}...") # Increased length for debugging
        query_job = bq_client.query(sql, job_config=job_config)
        results = query_job.result()
        logger.info("Query executed successfully.")
        return results
    except NotFound as e:
        logger.warning(f"Resource not found during query execution: {e}. Query: {sql[:150]}...")
        return None
    except BadRequest as e:
         # Sometimes "Not found" comes as BadRequest for tables/models used in queries
        if "Not found: Table" in str(e) or "Not found: Model" in str(e) or "Not found: Connection" in str(e):
             logger.warning(f"Resource (Table/Model/Connection) not found (BadRequest): {e}. Query: {sql[:150]}...")
             return None
        else:
            logger.error(f"BigQuery BadRequest error executing query: {e}", exc_info=True)
            logger.error(f"Query: {sql}")
            if hasattr(e, 'errors'): logger.error(f"  Errors: {e.errors}")
            raise # Re-raise other BadRequests
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

def setup_object_table():
    """Creates the Object Table referencing GCS images using the REUSED connection."""
    # (Keep this function as in the previous "Option 2" version)
    full_table_id = f"{GCP_PROJECT_ID}.{BQ_DATASET_ID}.{OBJECT_TABLE_ID}"
    logger.info(f"Ensuring Object Table {full_table_id} exists...")
    logger.warning(f"This setup assumes the service account for connection '{REUSED_CONNECTION_NAME}' has 'roles/storage.objectViewer' granted.")
    try:
        bq_client.get_table(full_table_id)
        logger.info(f"Object table {full_table_id} already exists.")
        return
    except NotFound:
        logger.info(f"Object table {full_table_id} not found, creating...")
        sql = f"""
        CREATE OR REPLACE EXTERNAL TABLE `{GCP_PROJECT_ID}.{BQ_DATASET_ID}.{OBJECT_TABLE_ID}`
        WITH CONNECTION `{REUSED_CONNECTION_NAME}`
        OPTIONS (
            object_metadata = 'SIMPLE',
            uris = ['{GCS_LOGOS_URI}', '{GCS_HEADSHOTS_URI}']
        );
        """
        execute_bq_query(sql)
    except Exception as e:
         logger.error(f"Error checking/creating object table {full_table_id}: {e}", exc_info=True)
         raise

def setup_remote_embedding_model():
    """Creates the remote model pointing to the multimodal endpoint."""
    # (Keep this function as in the previous "Option 2" version)
    full_model_id = f"{GCP_PROJECT_ID}.{BQ_DATASET_ID}.{EMBEDDING_MODEL_ID}"
    logger.info(f"Ensuring Remote Model {full_model_id} exists...")
    logger.warning(f"This setup assumes the service account for connection '{REUSED_CONNECTION_NAME}' has 'roles/aiplatform.user' granted and Vertex AI API is enabled.")
    try:
        bq_client.get_model(full_model_id)
        logger.info(f"Remote model {full_model_id} already exists.")
    except NotFound:
        logger.info(f"Remote model {full_model_id} not found, creating...")
        sql = f"""
        CREATE OR REPLACE MODEL `{GCP_PROJECT_ID}.{BQ_DATASET_ID}.{EMBEDDING_MODEL_ID}`
        REMOTE WITH CONNECTION `{REUSED_CONNECTION_NAME}`
        OPTIONS (endpoint = '{VERTEX_MULTIMODAL_ENDPOINT}');
        """
        execute_bq_query(sql)
    except Exception as e:
        logger.error(f"Error checking/creating remote model {full_model_id}: {e}", exc_info=True)
        raise

def setup_embedding_table():
    """Creates the table to store image embeddings and metadata (including entity_name).
       DELETES the table first to ensure the schema is updated.
    """
    full_table_id = f"{GCP_PROJECT_ID}.{BQ_DATASET_ID}.{EMBEDDING_TABLE_ID}"
    logger.info(f"Ensuring Embedding Table {full_table_id} exists with the correct schema...")

    # *** ADD THIS BLOCK TO DELETE THE TABLE FIRST ***
    try:
        logger.warning(f"Attempting to delete existing table {full_table_id} to ensure schema update...")
        bq_client.delete_table(full_table_id, not_found_ok=True) # not_found_ok=True prevents error if it doesn't exist
        logger.info(f"Table {full_table_id} deleted or did not exist.")
    except Exception as e:
        logger.error(f"Error attempting to delete table {full_table_id} (continuing to create): {e}", exc_info=True)
    # *** END BLOCK TO DELETE TABLE ***

    # Schema definition remains the same
    schema = [
        bigquery.SchemaField("image_uri", "STRING", mode="REQUIRED", description="GCS URI of the image"),
        bigquery.SchemaField("image_type", "STRING", mode="NULLABLE", description="Type of image: 'logo' or 'headshot'"),
        bigquery.SchemaField("entity_id", "STRING", mode="NULLABLE", description="Team ID/Name (from logo name) or Player ID (from headshot name)"),
        bigquery.SchemaField("entity_name", "STRING", mode="NULLABLE", description="Player Name (from metadata table) or Team Name (parsed)"), # <-- The important column
        bigquery.SchemaField("embedding", "FLOAT64", mode="REPEATED", description=f"Multimodal embedding vector ({EMBEDDING_DIMENSIONALITY} dimensions)"),
        bigquery.SchemaField("last_updated", "TIMESTAMP", mode="NULLABLE"),
    ]
    table = bigquery.Table(full_table_id, schema=schema)
    table.description = "Stores multimodal embeddings for MLB logos and player headshots"

    try:
        # Now create_table will definitely create it with the schema above
        bq_client.create_table(table) # Remove exists_ok=True if delete works reliably, or keep it as safety
        logger.info(f"Table {full_table_id} created with the latest schema.")
    except Conflict:
         logger.info(f"Table {full_table_id} already exists (potentially delete failed?). Check schema manually if errors persist.")
    except Exception as e:
        logger.error(f"Failed to create embedding table {full_table_id}: {e}", exc_info=True)
        raise

# --- New Functions for Player Metadata ---
def setup_player_metadata_table():
    """Creates the table to store player ID and name mappings."""
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
    logger.info("Fetching all team rosters to update player metadata...")
    all_players_data = []
    for team_name, team_id in TEAMS.items():
        logger.debug(f"Fetching roster for {team_name.replace('_', ' ').title()} (ID: {team_id})...")
        roster_data = call_mlb_roster_api(team_id, MLB_API_SEASON) # Use the imported/defined function
        if roster_data and 'roster' in roster_data:
            for player in roster_data['roster']:
                person = player.get('person', {})
                player_id = person.get('id')
                player_name = person.get('fullName')
                if player_id and player_name:
                    all_players_data.append({'player_id': player_id, 'player_name': player_name})
        else:
            logger.warning(f"Could not retrieve or parse roster for team {team_id}.")
        time.sleep(0.2) # Small sleep between API calls if not rate limited

    if not all_players_data:
        logger.error("No player data fetched. Cannot populate player metadata table.")
        return

    # Create DataFrame and remove duplicates
    players_df = pd.DataFrame(all_players_data)
    players_df = players_df.drop_duplicates(subset=['player_id'])
    players_df['player_id'] = pd.to_numeric(players_df['player_id'], errors='coerce').astype('Int64') # Ensure correct type
    players_df = players_df.dropna(subset=['player_id']) # Remove rows where ID couldn't be parsed
    players_df['last_updated'] = pd.Timestamp.now(tz='UTC')

    if players_df.empty:
        logger.warning("Player DataFrame is empty after cleaning. Cannot populate player metadata table.")
        return

    logger.info(f"Found {len(players_df)} unique players to upsert into metadata table.")

    # Upsert using MERGE statement (more robust than load + delete/insert)
    # We need to upload the DataFrame to a temporary table first for MERGE source
    temp_table_id = f"{PLAYER_METADATA_TABLE_ID}_temp_{int(time.time())}"
    full_temp_table_id = f"{GCP_PROJECT_ID}.{BQ_DATASET_ID}.{temp_table_id}"
    target_table_id = f"{GCP_PROJECT_ID}.{BQ_DATASET_ID}.{PLAYER_METADATA_TABLE_ID}"

    try:
        # Configure load job
        job_config = bigquery.LoadJobConfig(
            schema=[ # Define schema explicitly for temp table
                 bigquery.SchemaField("player_id", "INTEGER"),
                 bigquery.SchemaField("player_name", "STRING"),
                 bigquery.SchemaField("last_updated", "TIMESTAMP"),
            ],
            write_disposition="WRITE_TRUNCATE", # Overwrite temp table each time
        )
        # Load DataFrame to temporary table
        logger.info(f"Loading player data to temporary table {full_temp_table_id}...")
        load_job = bq_client.load_table_from_dataframe(players_df, full_temp_table_id, job_config=job_config)
        load_job.result() # Wait for completion
        logger.info(f"Loaded {load_job.output_rows} rows to temporary table.")

        # Execute MERGE statement
        merge_sql = f"""
        MERGE `{target_table_id}` T
        USING `{full_temp_table_id}` S
        ON T.player_id = S.player_id
        WHEN MATCHED THEN
            UPDATE SET T.player_name = S.player_name, T.last_updated = S.last_updated
        WHEN NOT MATCHED THEN
            INSERT (player_id, player_name, last_updated) VALUES(S.player_id, S.player_name, S.last_updated);
        """
        logger.info("Executing MERGE statement to update player metadata...")
        merge_job = bq_client.query(merge_sql)
        merge_job.result() # Wait for merge to complete
        logger.info("Player metadata table successfully updated.")

    except Exception as e:
        logger.error(f"Error during player metadata upsert: {e}", exc_info=True)
    finally:
        # Clean up temporary table
        try:
            logger.info(f"Deleting temporary table {full_temp_table_id}...")
            bq_client.delete_table(full_temp_table_id, not_found_ok=True)
        except Exception as e:
            logger.error(f"Error deleting temporary table {full_temp_table_id}: {e}", exc_info=True)

def generate_and_store_embeddings(batch_size: int = 20): # batch_size is now unused
    """Generates embeddings using ML.GENERATE_EMBEDDING and stores them, joining player names.
       Passes the FULL OBJECT TABLE directly to ML.GENERATE_EMBEDDING.
       Applies anti-duplication check *after* embedding generation, before insert.
    """
    object_table_ref = f"`{GCP_PROJECT_ID}.{BQ_DATASET_ID}.{OBJECT_TABLE_ID}`"
    embedding_table_ref = f"`{GCP_PROJECT_ID}.{BQ_DATASET_ID}.{EMBEDDING_TABLE_ID}`"
    model_ref = f"`{GCP_PROJECT_ID}.{BQ_DATASET_ID}.{EMBEDDING_MODEL_ID}`"
    player_meta_ref = f"`{GCP_PROJECT_ID}.{BQ_DATASET_ID}.{PLAYER_METADATA_TABLE_ID}`"

    # --- Count checks (optional, mainly for logging) ---
    count_existing_sql = f"SELECT count(*) as existing_count FROM {embedding_table_ref}"
    logger.info("Checking count of existing embeddings...")
    count_existing_result = execute_bq_query(count_existing_sql)
    existing_rows = 0
    if count_existing_result:
        try: existing_rows = list(count_existing_result)[0].existing_count
        except IndexError: pass
    logger.info(f"Found {existing_rows} existing embeddings.")

    count_sql = f"SELECT count(*) as total_count FROM {object_table_ref}"
    logger.info("Attempting to count objects in object table...")
    count_result = execute_bq_query(count_sql)
    if not count_result:
        logger.error("Could not get count from object table. Aborting embedding generation.")
        return
    total_rows = list(count_result)[0].total_count
    logger.info(f"Found {total_rows} objects in {object_table_ref}.")
    if total_rows == 0:
        logger.info("Object table is empty, skipping embedding generation.")
        return

    logger.info(f"Starting embedding generation and insertion query (will process all {total_rows} objects)...")
    processed_in_query = 0
    error_count = 0

    # --- Single Query - Embed ALL objects, then filter ---
    # Pass the full object table directly to ML.GENERATE_EMBEDDING
    generate_sql = f"""
    INSERT INTO {embedding_table_ref} (image_uri, image_type, entity_id, entity_name, embedding, last_updated)
    WITH EmbeddingsAllRaw AS (
        -- Generate embeddings for ALL objects in the object table
        SELECT uri, content_type, ml_generate_embedding_status, ml_generate_embedding_result
        FROM ML.GENERATE_EMBEDDING(
            MODEL {model_ref},
            TABLE {object_table_ref} -- <<< Pass the OBJECT TABLE directly
        )
        -- Note: We filter for 'OK' status later if needed, process all results first
    ),
    EmbeddingsOk AS (
      -- Filter for successful embeddings before parsing/joining
      SELECT uri, ml_generate_embedding_result
      FROM EmbeddingsAllRaw
      WHERE ml_generate_embedding_status = 'OK'
    ),
    ParsedMeta AS (
        -- Parse metadata from the URIs we successfully got embeddings for
        SELECT
            E.uri, E.ml_generate_embedding_result,
            CASE WHEN E.uri LIKE '%/logos/%' THEN 'logo' WHEN E.uri LIKE '%/headshots/%' THEN 'headshot' ELSE 'unknown' END AS image_type,
            CASE
                WHEN E.uri LIKE '%/logos/%' THEN COALESCE(REGEXP_EXTRACT(E.uri, r'logos/.*?(\d+).*\.png$'), REGEXP_EXTRACT(E.uri, r'logos/mlb-([a-z0-9-]+(?:-[a-z0-9-]+)*)-logo\.png$'))
                WHEN E.uri LIKE '%/headshots/%' THEN CAST(REGEXP_EXTRACT(E.uri, r'headshots/headshot_(\d+)\.\w+$') AS STRING)
                ELSE NULL
            END AS parsed_entity_id
        FROM EmbeddingsOk E -- Process only OK embeddings
    ),
    ParsedMetaForJoin AS (
        SELECT *, SAFE_CAST(parsed_entity_id AS INT64) as joinable_player_id
        FROM ParsedMeta WHERE image_type = 'headshot'
    ),
    FinalDataToInsert AS (
      -- Combine embeddings and metadata, joining for player names
      SELECT
          PM.uri AS image_uri, PM.image_type, PM.parsed_entity_id AS entity_id,
          CASE
              WHEN PM.image_type = 'headshot' AND PJ.joinable_player_id IS NOT NULL THEN P.player_name
              ELSE PM.parsed_entity_id
          END AS entity_name,
          PM.ml_generate_embedding_result AS embedding, CURRENT_TIMESTAMP() AS last_updated
      FROM ParsedMeta PM
      LEFT JOIN ParsedMetaForJoin PJ ON PM.uri = PJ.uri
      LEFT JOIN {player_meta_ref} P ON PJ.joinable_player_id = P.player_id
    )
    -- Final SELECT for INSERT, applying anti-duplication check HERE
    SELECT F.*
    FROM FinalDataToInsert F
    LEFT JOIN {embedding_table_ref} Existing ON F.image_uri = Existing.image_uri
    WHERE Existing.image_uri IS NULL; -- Only insert if URI doesn't exist in target table
    """

    logger.info("Executing single query to embed all objects and insert missing...")
    try:
        job_config = bigquery.QueryJobConfig(priority=bigquery.QueryPriority.BATCH)
        job = bq_client.query(generate_sql, job_config=job_config)
        job.result(timeout=3600) # Allow up to 1 hour
        logger.info(f"Embedding generation and insertion job completed. Job ID: {job.job_id}")
        if job.num_dml_affected_rows is not None:
            processed_in_query = job.num_dml_affected_rows
            logger.info(f" -> Inserted {processed_in_query} new rows.")
        else:
             logger.info(f" -> Insertion complete (affected rows count unavailable). Check table count manually.")

    except Exception as e:
        error_count += 1
        logger.error(f"Error executing the main embedding/insertion query: {e}", exc_info=True)
        # Log the failed SQL (it's long, but helpful for debugging)
        logger.error(f"Failed Query SQL:\n{generate_sql}")

    logger.info(f"Finished embedding generation process. Inserted approximately {processed_in_query} new embeddings.")
    if error_count > 0:
        logger.warning(f"Encountered {error_count} errors during the embedding query execution.")


def setup_vector_index():
    """Creates the Vector Index on the embedding table."""
    # (Keep this function as in the previous "Option 2" version)
    embedding_table_fqn = f"`{GCP_PROJECT_ID}.{BQ_DATASET_ID}.{EMBEDDING_TABLE_ID}`"
    index_fqn = f"`{GCP_PROJECT_ID}.{BQ_DATASET_ID}`.{VECTOR_INDEX_ID}"
    index_check_sql = f"""
    SELECT index_name
    FROM `{GCP_PROJECT_ID}.{BQ_DATASET_ID}`.INFORMATION_SCHEMA.VECTOR_INDEXES
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


# --- Search Function (Updated for entity_name) ---

def search_similar_images(
    query_text: str,
    top_k: int = 5,
    filter_image_type: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Performs vector search, returning entity_name."""
    logger.info(f"Performing vector search for: '{query_text}', top_k={top_k}, filter='{filter_image_type}'")
    embedding_table_ref = f"`{GCP_PROJECT_ID}.{BQ_DATASET_ID}.{EMBEDDING_TABLE_ID}`"
    model_ref = f"`{GCP_PROJECT_ID}.{BQ_DATASET_ID}.{EMBEDDING_MODEL_ID}`"
    results = []

    try:
        # 1. Generate query embedding
        query_embedding_sql = f"""
        SELECT ml_generate_embedding_result
        FROM ML.GENERATE_EMBEDDING( MODEL {model_ref}, (SELECT '{query_text}' AS content) )
        WHERE ml_generate_embedding_status = 'OK' LIMIT 1;
        """
        embedding_result = execute_bq_query(query_embedding_sql)
        if not embedding_result:
            logger.error("Failed to generate embedding for the search query text (Step 1). Check model and connection.")
            return []
        query_embedding_list = list(embedding_result)
        if not query_embedding_list or not query_embedding_list[0].ml_generate_embedding_result:
            logger.error("Embedding generation for search query returned no result (Step 1).")
            return []
        query_embedding = query_embedding_list[0].ml_generate_embedding_result
        query_embedding_str = f"[{','.join(map(str, query_embedding))}]"

        # 2. Perform Vector Search
        filter_str = ""
        if filter_image_type:
            escaped_filter_type = filter_image_type.replace("'", "\\'")
            filter_str = f', options => \'{{"filter": "image_type=\'{escaped_filter_type}\'"}}\''

        # *** UPDATED SELECT LIST ***
        vector_search_sql = f"""
        SELECT
            base.image_uri,
            base.image_type,
            base.entity_id,
            base.entity_name, -- <-- Added entity_name
            distance
        FROM
            VECTOR_SEARCH(
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
            # This might happen if the index isn't ready yet or no matches
            logger.warning("Vector search returned no results (check index status or query).")

    except Exception as e:
        logger.error(f"Error during vector search for '{query_text}': {e}", exc_info=True)
        return []

    return results


# --- Main Execution Flow ---
if __name__ == "__main__":
    logger.info("--- Starting MLB Image Embedding Pipeline Setup (Using Reused Connection + Player Names) ---")
    logger.warning(f"ASSUMING Connection '{REUSED_CONNECTION_NAME}' service account has 'roles/storage.objectViewer' AND 'roles/aiplatform.user' access.")
    logger.warning("ASSUMING Vertex AI API is enabled.")
    full_start_time = time.time()

    # --- Step 0: Setup and Populate Player Metadata Table ---
    try:
        logger.info("\n=== Step 0: Setting up Player Metadata Table ===")
        meta_start_time = time.time()
        setup_player_metadata_table()
        # Decide whether to run populate every time or less frequently
        # For now, run it each time to ensure it's up-to-date for the demo
        populate_player_metadata()
        logger.info(f"Player metadata setup/update took {time.time() - meta_start_time:.2f} seconds.")
    except Exception as e:
        logger.critical(f"Failed during Player Metadata setup: {e}. Aborting.", exc_info=True)
        # Decide if you want to proceed without player names or stop
        exit(1) # Stop if metadata fails

    # --- Step 1: Setup BQ Resources (Object Table, Model, Embedding Table) ---
    try:
        logger.info("\n=== Step 1: Setting up Core BigQuery Resources ===")
        setup_start_time = time.time()
        setup_object_table()          # Needs GCS connection with storage.objectViewer
        setup_remote_embedding_model() # Needs Vertex connection with aiplatform.user + API enabled
        setup_embedding_table()       # Just needs BQ access
        logger.info(f"Core resource setup took {time.time() - setup_start_time:.2f} seconds.")
    except Exception as e:
        logger.critical(f"Failed during core BigQuery resource setup: {e}. Aborting.", exc_info=True)
        exit(1)

    # --- Step 2: Generate and Store Embeddings ---
    try:
        logger.info("\n=== Step 2: Generating and Storing Embeddings (with Player Name join) ===")
        embed_start_time = time.time()
        generate_and_store_embeddings(batch_size=25) # Reduced batch size slightly
        logger.info(f"Embedding generation took {time.time() - embed_start_time:.2f} seconds.")
    except Exception as e:
        logger.error(f"Failed during embedding generation: {e}.", exc_info=True)
        logger.warning("Proceeding to index creation despite potential embedding errors.")

    # --- Step 3: Setup Vector Index ---
    try:
        logger.info("\n=== Step 3: Setting up Vector Index ===")
        index_start_time = time.time()
        setup_vector_index()
        logger.info(f"Vector index setup initiated took {time.time() - index_start_time:.2f} seconds (building happens async).")
    except Exception as e:
        logger.error(f"Failed during vector index setup: {e}.", exc_info=True)

    # --- Step 4: Example Search ---
    try:
        logger.info("\n=== Step 4: Example Vector Search ===")
        search_start_time = time.time()
        logger.info("Waiting 90 seconds before example search (index building takes time, especially after initial load)...")
        time.sleep(90) # Increased wait time

        search_query_logo = "Arizona Diamondbacks logo"
        logo_results = search_similar_images(search_query_logo, top_k=3, filter_image_type='logo')
        print(f"\nSearch Results for '{search_query_logo}':")
        # Use default=str for potential non-serializable types like Timestamps if they sneak in
        print(json.dumps(logo_results, indent=2, default=str))

        search_query_headshot = "player Mookie Betts"
        headshot_results = search_similar_images(search_query_headshot, top_k=3, filter_image_type='headshot')
        print(f"\nSearch Results for '{search_query_headshot}':")
        print(json.dumps(headshot_results, indent=2, default=str))

        # Example without filter
        search_query_generic = "baseball player hitting"
        generic_results = search_similar_images(search_query_generic, top_k=5)
        print(f"\nSearch Results for '{search_query_generic}' (mixed types):")
        print(json.dumps(generic_results, indent=2, default=str))


        logger.info(f"Example search execution took {time.time() - search_start_time:.2f} seconds (excluding wait time).")
    except Exception as e:
        logger.error(f"Failed during example search: {e}", exc_info=True)


    logger.info(f"\n--- MLB Image Embedding Pipeline Setup Finished ---")
    logger.info(f"Total script execution time: {time.time() - full_start_time:.2f} seconds")