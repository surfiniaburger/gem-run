import os
import json
import time
import logging
import requests
import pandas as pd
import numpy as np

from google.cloud import bigquery
import vertexai
from vertexai.language_models import TextEmbeddingModel
from vertexai.vision_models import MultiModalEmbeddingModel, Video
from vertexai.generative_models import GenerativeModel, Part

# --- Configuration ---
PROJECT_ID = "silver-455021"  # Replace with your Project ID
LOCATION = "us-central1"      # Replace with your desired location
BQ_DATASET = "mlb_hackathon_rag" # Replace with your BQ dataset name
BQ_TABLE = "play_by_play_embeddings" # Replace with your desired BQ table name
VIDEO_GCS_BUCKET = "gcp-mlb-hackathon-2025-videos" # IMPORTANT: Replace with YOUR GCS bucket name where videos are stored
EMBEDDING_MODEL_TEXT = "text-embedding-005" # Or the latest stable version
EMBEDDING_MODEL_MULTIMODAL = "multimodalembedding@001"
GEMINI_MODEL = "gemini-2.5-pro-exp-03-25" # Or another suitable Gemini model

# Home run data URL
HR_CSV_URL = 'https://storage.googleapis.com/gcp-mlb-hackathon-2025/datasets/2024-mlb-homeruns.csv'

# Embedding Dimensions (match your models)
TEXT_EMBEDDING_DIM = 768 # For text-embedding-005
VIDEO_EMBEDDING_DIM = 1408 # For multimodalembedding@001

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Initialization ---
try:
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    bq_client = bigquery.Client(project=PROJECT_ID)
    text_embedding_model = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL_TEXT)
    multimodal_embedding_model = MultiModalEmbeddingModel.from_pretrained(EMBEDDING_MODEL_MULTIMODAL)
    generative_model = GenerativeModel(GEMINI_MODEL)
    logging.info("Vertex AI and BigQuery clients initialized.")
except Exception as e:
    logging.error(f"Initialization failed: {e}", exc_info=True)
    exit()

# --- Helper Functions ---

def fetch_mlb_api(url: str) -> dict | None:
    """Fetches data from MLB Stats API URL and parses JSON."""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching {url}: {e}")
        return None
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON from {url}: {e}")
        return None

def get_text_embedding(text: str) -> list[float] | None:
    """Generates text embedding using the Vertex AI model."""
    if not text:
        return None
    try:
        embeddings = text_embedding_model.get_embeddings([text])
        # Check if embeddings were successfully generated
        if embeddings and embeddings[0].values:
             # Ensure the embedding has the correct dimension
             embedding_values = embeddings[0].values
             if len(embedding_values) == TEXT_EMBEDDING_DIM:
                 return embedding_values
             else:
                 logging.warning(f"Text embedding dimension mismatch: Expected {TEXT_EMBEDDING_DIM}, Got {len(embedding_values)}. Padding/truncating (Not implemented here).")
                 # Handle mismatch - e.g., pad with zeros or truncate. For simplicity, returning None here.
                 return None # Or implement padding/truncation if required by BQ schema
        else:
             logging.warning(f"No embedding values returned for text: '{text[:50]}...'")
             return None

    except Exception as e:
        logging.error(f"Error getting text embedding: {e}", exc_info=True)
        return None

def get_video_embedding(video_gcs_uri: str) -> list[float] | None:
    """Generates video embedding using the Vertex AI model."""
    if not video_gcs_uri or not video_gcs_uri.startswith("gs://"):
        logging.warning(f"Invalid GCS URI for video embedding: {video_gcs_uri}")
        return None
    try:
        video = Video.load_from_file(video_gcs_uri)
        embeddings = multimodal_embedding_model.get_embeddings(
            video=video,
            dimension=VIDEO_EMBEDDING_DIM,
            # video_segment_config can be added here if needed
        )
        # Use the embedding from the first segment as a representative embedding
        if embeddings.video_embeddings and embeddings.video_embeddings[0].embedding:
             embedding_values = embeddings.video_embeddings[0].embedding
             if len(embedding_values) == VIDEO_EMBEDDING_DIM:
                 return embedding_values
             else:
                  logging.warning(f"Video embedding dimension mismatch: Expected {VIDEO_EMBEDDING_DIM}, Got {len(embedding_values)}. Padding/truncating (Not implemented here).")
                  # Handle mismatch - e.g., pad with zeros or truncate. For simplicity, returning None here.
                  return None # Or implement padding/truncation if required by BQ schema
        else:
            logging.warning(f"No embedding values returned for video: {video_gcs_uri}")
            return None
    except Exception as e:
        # Catch specific exceptions like file not found if possible
        logging.error(f"Error getting video embedding for {video_gcs_uri}: {e}", exc_info=True)
        return None

def map_https_to_gcs(https_url: str, bucket_name: str) -> str | None:
    """Attempts to map an HTTPS URL (from HR dataset) to a GCS URI."""
    if not https_url or not isinstance(https_url, str):
        return None
    try:
        # Extract filename from URL path
        filename = https_url.split('/')[-1]
        if filename:
            return f"gs://{bucket_name}/{filename}"
        else:
            return None
    except Exception as e:
        logging.error(f"Error mapping HTTPS URL {https_url} to GCS: {e}")
        return None

def setup_bigquery_table():
    """Creates the BigQuery table with VECTOR types if it doesn't exist."""
    table_id = f"{PROJECT_ID}.{BQ_DATASET}.{BQ_TABLE}"
    schema = [
        bigquery.SchemaField("game_pk", "INTEGER", mode="REQUIRED"),
        bigquery.SchemaField("play_guid", "STRING", mode="REQUIRED"), # Using play GUID as a unique ID within a game
        bigquery.SchemaField("inning", "INTEGER"),
        bigquery.SchemaField("half_inning", "STRING"),
        bigquery.SchemaField("at_bat_index", "INTEGER"),
        bigquery.SchemaField("play_description", "STRING"),
        bigquery.SchemaField("play_details_text", "STRING"), # Combined details
        bigquery.SchemaField("away_score", "INTEGER"),
        bigquery.SchemaField("home_score", "INTEGER"),
        bigquery.SchemaField("batter_id", "INTEGER"),
        bigquery.SchemaField("batter_name", "STRING"),
        bigquery.SchemaField("pitcher_id", "INTEGER"),
        bigquery.SchemaField("pitcher_name", "STRING"),
        bigquery.SchemaField("video_url", "STRING"),
        bigquery.SchemaField("video_gcs_uri", "STRING"),
        # VECTOR types require specifying dimensions
        bigquery.SchemaField(
            "text_embedding",
            "VECTOR",
            mode="NULLABLE",
            fields=(bigquery.SchemaField("dimension", "INT64", mode="REQUIRED"),)
        ),
         bigquery.SchemaField(
             "video_embedding",
             "VECTOR",
             mode="NULLABLE",
             fields=(bigquery.SchemaField("dimension", "INT64", mode="REQUIRED"),)
         ),
         bigquery.SchemaField("processing_timestamp", "TIMESTAMP", mode="REQUIRED"),
    ]

    # Add vector dimension constraints during table creation (or ALTER TABLE)
    table = bigquery.Table(table_id, schema=schema)
    table.clustering_fields = ["game_pk"] # Example clustering

    # Example of setting vector options (dimensions)
    # Note: Direct support in client library might vary. May need raw DDL.
    # This part is illustrative; precise syntax might depend on library version/BQ features.
    # You might need to create the table first, then ALTER TABLE to add vector options.
    # DDL Example:
    # CREATE TABLE `project.dataset.table` ( ... text_embedding VECTOR(768), video_embedding VECTOR(1408) ... )

    try:
        bq_client.create_table(table, exists_ok=True) # exists_ok=True prevents error if table exists
        logging.info(f"Table {table_id} created or already exists.")
        # Consider adding vector index creation here or separately using DDL
        # Example DDL for Index:
        # CREATE VECTOR INDEX my_text_index ON `project.dataset.table`(text_embedding) OPTIONS(distance_type='COSINE', index_type='IVF')
    except Exception as e:
        logging.error(f"Error creating BigQuery table {table_id}: {e}", exc_info=True)
        raise # Re-raise the exception to stop execution if table setup fails

def load_to_bigquery(rows: list[dict]):
    """Loads processed data rows into the BigQuery table."""
    if not rows:
        logging.info("No rows to load into BigQuery.")
        return

    table_id = f"{PROJECT_ID}.{BQ_DATASET}.{BQ_TABLE}"
    # Add vector dimensions to the data before insertion
    formatted_rows = []
    for row in rows:
         new_row = row.copy() # Avoid modifying the original list
         if new_row.get("text_embedding"):
             new_row["text_embedding"] = {
                 "values": new_row["text_embedding"],
                 "dimension": TEXT_EMBEDDING_DIM
             }
         if new_row.get("video_embedding"):
              new_row["video_embedding"] = {
                  "values": new_row["video_embedding"],
                  "dimension": VIDEO_EMBEDDING_DIM
              }
         formatted_rows.append(new_row)

    try:
        # Use insert_rows_json for streaming-like insertion
        errors = bq_client.insert_rows_json(table_id, formatted_rows)
        if errors == []:
            logging.info(f"Successfully loaded {len(rows)} rows to {table_id}")
        else:
            logging.error(f"Errors encountered while loading data to {table_id}: {errors}")
    except Exception as e:
        logging.error(f"Error loading data to BigQuery table {table_id}: {e}", exc_info=True)


# --- Main Processing Logic ---

def process_game_data(game_pk: int, hr_df: pd.DataFrame):
    """Fetches game data, processes plays, generates embeddings, and returns rows for BQ."""
    logging.info(f"Processing game: {game_pk}")
    single_game_feed_url = f'https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live'
    game_data = fetch_mlb_api(single_game_feed_url)

    if not game_data or 'liveData' not in game_data or 'plays' not in game_data['liveData'] or 'allPlays' not in game_data['liveData']['plays']:
        logging.error(f"Could not retrieve valid play data for game {game_pk}")
        return []

    all_plays = game_data['liveData']['plays']['allPlays']
    processed_rows = []
    processing_ts = bigquery.ScalarQueryParameter("processing_ts", "TIMESTAMP", time.strftime('%Y-%m-%d %H:%M:%S UTC'))


    logging.info(f"Found {len(all_plays)} plays for game {game_pk}.")

    # Rate limiting - adjust as needed
    requests_per_minute = 150 # Example limit for embedding APIs
    delay = 60.0 / requests_per_minute

    for i, play in enumerate(all_plays):
        logging.debug(f"Processing play {i+1}/{len(all_plays)} (Index: {play.get('about', {}).get('atBatIndex')})")

        # --- Extract Text Data ---
        play_result = play.get('result', {})
        play_about = play.get('about', {})
        play_matchup = play.get('matchup', {})
        play_events = play.get('playEvents', []) # List of pitch events etc.

        description = play_result.get('description', '')
        inning = play_about.get('inning')
        half_inning = play_about.get('halfInning')
        at_bat_index = play_about.get('atBatIndex')
        away_score = play_result.get('awayScore')
        home_score = play_result.get('homeScore')
        batter = play_matchup.get('batter', {})
        pitcher = play_matchup.get('pitcher', {})

        # Create a more detailed text representation for embedding
        details_text = f"Inning: {inning} ({half_inning}). Score: {away_score}-{home_score}. "
        details_text += f"Batter: {batter.get('fullName', 'N/A')}. Pitcher: {pitcher.get('fullName', 'N/A')}. "
        details_text += f"Result: {description}"

        # --- Find Associated Video ---
        video_url = None
        video_gcs_uri = None
        play_guid = None # Need a reliable unique ID for the play within the game

        # Find the *last* event in the play to get the final playId/GUID
        if play_events:
            last_event = play_events[-1]
            play_guid = last_event.get('playId') # playId seems to be a GUID

        if play_guid:
            # Match with home run data
            match = hr_df[hr_df['play_id'] == play_guid]
            if not match.empty:
                video_url = match.iloc[0]['video']
                video_gcs_uri = match.iloc[0]['gcs_uri'] # Use the pre-mapped GCS URI

        # --- Generate Embeddings ---
        logging.debug("Generating text embedding...")
        text_emb = get_text_embedding(details_text)
        time.sleep(delay) # Respect rate limits

        video_emb = None
        if video_gcs_uri:
            logging.debug(f"Generating video embedding for: {video_gcs_uri}")
            video_emb = get_video_embedding(video_gcs_uri)
            time.sleep(delay) # Respect rate limits
        else:
             logging.debug("No video GCS URI found for this play.")


        # --- Prepare Row for BigQuery ---
        if play_guid: # Only add if we have a play ID
            row = {
                "game_pk": game_pk,
                "play_guid": play_guid,
                "inning": inning,
                "half_inning": half_inning,
                "at_bat_index": at_bat_index,
                "play_description": description,
                "play_details_text": details_text,
                "away_score": away_score,
                "home_score": home_score,
                "batter_id": batter.get('id'),
                "batter_name": batter.get('fullName'),
                "pitcher_id": pitcher.get('id'),
                "pitcher_name": pitcher.get('fullName'),
                "video_url": video_url,
                "video_gcs_uri": video_gcs_uri,
                "text_embedding": text_emb,
                "video_embedding": video_emb,
                "processing_timestamp": processing_ts.value, # Get value from query param
            }
            processed_rows.append(row)
        else:
             logging.warning(f"Skipping play at index {i} due to missing playId/GUID.")


    logging.info(f"Finished processing {len(processed_rows)} plays for game {game_pk}.")
    return processed_rows

# --- Retrieval Functions ---

def search_bigquery_text(query_text: str, top_n: int = 5) -> pd.DataFrame:
    """Searches BigQuery for plays with similar text embeddings."""
    logging.info(f"Searching BigQuery text for: '{query_text}'")
    query_embedding = get_text_embedding(query_text)
    if not query_embedding:
        logging.error("Could not generate embedding for query text.")
        return pd.DataFrame()

    table_id = f"`{PROJECT_ID}.{BQ_DATASET}.{BQ_TABLE}`"
    # Convert list to string format BQ expects for array literals
    query_embedding_str = str(query_embedding)

    sql = f"""
    SELECT
        base.game_pk,
        base.play_guid,
        base.inning,
        base.half_inning,
        base.play_description,
        base.play_details_text,
        base.video_url,
        base.video_gcs_uri,
        distance
    FROM
        VECTOR_SEARCH(
            TABLE {table_id},            -- Table to search
            'text_embedding',            -- Column with vectors
            (SELECT {query_embedding_str} AS embedding), -- Query vector
            top_k => {top_n},
            distance_type => 'COSINE'     -- Or 'EUCLIDEAN'/'DOT_PRODUCT'
            -- options => '{{"use_brute_force":true}}' -- Optional: for tables without index
        ) base
    ORDER BY distance ASC; -- COSINE distance is lower for more similar items
    """

    try:
        logging.debug(f"Executing BigQuery SQL: {sql[:200]}...") # Log truncated query
        results_df = bq_client.query(sql).to_dataframe()
        logging.info(f"Found {len(results_df)} text results in BigQuery.")
        return results_df
    except Exception as e:
        logging.error(f"BigQuery text search failed: {e}", exc_info=True)
        return pd.DataFrame()

def search_bigquery_video(query_text: str, top_n: int = 5) -> pd.DataFrame:
    """Searches BigQuery for plays with video embeddings similar to query text."""
    logging.info(f"Searching BigQuery video embeddings based on text: '{query_text}'")
    # We use the *text* embedding of the query to search against *video* embeddings
    # This works because the multimodal model maps text and visuals to the same space
    query_embedding = get_text_embedding(query_text) # Use text embedding model for query
    if not query_embedding:
        logging.error("Could not generate embedding for query text.")
        return pd.DataFrame()

    table_id = f"`{PROJECT_ID}.{BQ_DATASET}.{BQ_TABLE}`"
    query_embedding_str = str(query_embedding)

    # IMPORTANT: Ensure the query embedding dimension matches the VIDEO_EMBEDDING_DIM
    # If text_embedding_model and multimodal_embedding_model have different output
    # dimensions, this search won't work directly. You'd need a model that outputs
    # embeddings in the same dimension for both text and video for this specific cross-modal search.
    # Assuming multimodalembedding@001 can produce text embeddings compatible with its video embeddings.
    # Let's re-embed the text query using the multimodal model for compatibility:
    try:
         mm_query_emb_obj = multimodal_embedding_model.get_embeddings(
             contextual_text=query_text,
             dimension=VIDEO_EMBEDDING_DIM
         )
         mm_query_embedding = mm_query_emb_obj.text_embedding
         if not mm_query_embedding or len(mm_query_embedding) != VIDEO_EMBEDDING_DIM:
              logging.error(f"Could not generate compatible multimodal text embedding (Dim: {VIDEO_EMBEDDING_DIM}).")
              return pd.DataFrame()
         query_embedding_str = str(mm_query_embedding)
         logging.info("Using multimodal model to generate query text embedding for video search.")
    except Exception as e:
         logging.error(f"Error generating multimodal text embedding for query: {e}", exc_info=True)
         return pd.DataFrame()


    sql = f"""
    SELECT
        base.game_pk,
        base.play_guid,
        base.inning,
        base.half_inning,
        base.play_description,
        base.play_details_text,
        base.video_url,
        base.video_gcs_uri,
        distance
    FROM
        VECTOR_SEARCH(
            TABLE {table_id},
            'video_embedding', -- Search the video embedding column
            (SELECT {query_embedding_str} AS embedding), -- Use the text query embedding
            top_k => {top_n},
            distance_type => 'COSINE'
        ) base
    WHERE base.video_embedding IS NOT NULL -- Only consider rows with video embeddings
    ORDER BY distance ASC;
    """

    try:
        logging.debug(f"Executing BigQuery SQL: {sql[:200]}...")
        results_df = bq_client.query(sql).to_dataframe()
        logging.info(f"Found {len(results_df)} video results in BigQuery based on text query.")
        return results_df
    except Exception as e:
        logging.error(f"BigQuery video search failed: {e}", exc_info=True)
        return pd.DataFrame()


# --- Generation Function ---

def generate_rag_response(query: str, text_context: str, video_gcs_uris: list[str]) -> str:
    """Generates a response from Gemini using retrieved text and video context."""
    logging.info("Generating response with Gemini...")
    model_input_parts = [f"Question: {query}\n\nUse the following context to answer:\n"]

    # Add Text Context
    if text_context:
        model_input_parts.append("--- Text Context ---\n")
        model_input_parts.append(text_context)
        model_input_parts.append("\n")

    # Add Video Context
    if video_gcs_uris:
        model_input_parts.append("--- Video Context ---\n")
        for i, uri in enumerate(video_gcs_uris):
            try:
                video_part = Part.from_uri(uri, mime_type="video/mp4") # Assuming mp4
                model_input_parts.append(f"Video {i+1} ({uri.split('/')[-1]}):\n")
                model_input_parts.append(video_part)
                model_input_parts.append("\n")
            except Exception as e:
                logging.warning(f"Could not load video {uri} for Gemini: {e}")

    model_input_parts.append("\nAnswer:")

    # Check if any context was actually added
    if len(model_input_parts) <= 2: # Only the initial prompt and "Answer:"
         logging.warning("No valid context found to send to Gemini.")
         # Decide how to handle - return default message or query without context?
         # Example: Querying without context might be better than nothing.
         model_input_parts = [query]


    try:
        response = generative_model.generate_content(model_input_parts, stream=False) # Use stream=True for chunked response
        logging.info("Successfully generated response from Gemini.")
        return response.text
    except Exception as e:
        logging.error(f"Error generating response from Gemini: {e}", exc_info=True)
        return "Error: Could not generate response."


# --- Main Execution Example ---

if __name__ == "__main__":
    logging.info("Starting Multimodal RAG MLB Processing...")

    # 1. Setup BigQuery Table (Run once or ensure it exists)
    try:
        setup_bigquery_table()
    except Exception as e:
        logging.critical(f"Exiting due to BigQuery table setup failure: {e}")
        exit()

    # 2. Load and Prepare Home Run Data
    try:
        hr_df = pd.read_csv(HR_CSV_URL)
        # Map HTTPS URLs to GCS URIs - **Adjust bucket name!**
        hr_df['gcs_uri'] = hr_df['video'].apply(lambda url: map_https_to_gcs(url, VIDEO_GCS_BUCKET))
        logging.info(f"Loaded {len(hr_df)} home run records. Sample GCS URI mapping:")
        logging.info(hr_df[['play_id', 'video', 'gcs_uri']].head())
        # Filter out rows where mapping failed if necessary
        hr_df_filtered = hr_df.dropna(subset=['gcs_uri'])
        logging.info(f"{len(hr_df_filtered)} home runs have valid GCS URIs after mapping.")
    except Exception as e:
        logging.error(f"Failed to load or process home run data: {e}", exc_info=True)
        hr_df_filtered = pd.DataFrame() # Ensure it's an empty DataFrame if loading fails


    # 3. Process a Sample Game (Example: Last game from notebook 775296)
    #    Replace with logic to find games dynamically if needed
    sample_game_pk = 747066 # Braves vs Royals 9/28 walk-off
    if not hr_df_filtered.empty:
        game_rows = process_game_data(sample_game_pk, hr_df_filtered)
        load_to_bigquery(game_rows)
    else:
        logging.warning("Skipping game processing as home run data with GCS URIs is not available.")


    # --- Example RAG Query ---
    logging.info("\n--- Performing RAG Query ---")
    user_query = "Show me walk off home runs hit by catchers in late innings"

    # 4. Retrieve Context from BigQuery
    text_results = search_bigquery_text(user_query, top_n=3)
    video_results = search_bigquery_video(user_query, top_n=2) # Find videos related to the query text

    print("\n--- Text Search Results ---")
    print(text_results[['game_pk', 'play_guid', 'play_details_text', 'distance']].head())

    print("\n--- Video Search Results (based on text query) ---")
    print(video_results[['game_pk', 'play_guid', 'play_details_text', 'video_gcs_uri', 'distance']].head())

    # 5. Prepare Context for Gemini
    # Combine unique text descriptions
    all_text_context = pd.concat([text_results['play_details_text'], video_results['play_details_text']]).unique()
    combined_text = "\n---\n".join(all_text_context[:5]) # Limit context size

    # Get unique GCS URIs for videos
    video_uris = video_results['video_gcs_uri'].dropna().unique().tolist()[:2] # Limit number of videos

    # 6. Generate Response
    if combined_text or video_uris:
         final_response = generate_rag_response(user_query, combined_text, video_uris)
         print("\n--- Generated Answer ---")
         print(final_response)
    else:
         print("\nNo relevant text or video context found in BigQuery for the query.")


    logging.info("Multimodal RAG MLB Processing Finished.")