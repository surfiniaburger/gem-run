from datetime import datetime, date, UTC, timedelta
import logging
import google.cloud.logging
from google.cloud.logging.handlers import CloudLoggingHandler
from google.cloud import secretmanager
import pandas as pd
from google.cloud import bigquery
from typing import Dict, List, Tuple, Optional
from vertexai.language_models import TextEmbeddingModel
from google.cloud import aiplatform
import numpy as np
import requests
from ratelimit import limits, sleep_and_retry
from sklearn.feature_extraction.text import TfidfVectorizer
import sys
import os
from google.cloud import storage
import pickle


# --- Setup (Logging, BigQuery Client, Rate Limiting) ---

def setup_logging():
    """Sets up Google Cloud Logging."""
    client = google.cloud.logging.Client()
    handler = CloudLoggingHandler(client)
    logger = logging.getLogger('bigquery_vector_search_hybrid')
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    return logger

logger = setup_logging()

TEAMS = {
        'rangers': 140,
        'angels': 108,
        'astros': 117,
        'rays': 139,
        'blue_jays': 141,
        'yankees': 147,
        'orioles': 110,
        'red_sox': 111,
        'twins': 142,
        'white_sox': 145,
        'guardians': 114,
        'tigers': 116,
        'royals': 118,
        'padres': 135,
        'giants': 137,
        'diamondbacks': 109,
        'rockies': 115,
        'phillies': 143,
        'braves': 144,
        'marlins': 146,
        'nationals': 120,
        'mets': 121,
        'pirates': 134,
        'cardinals': 138,
        'brewers': 158,
        'cubs': 112,
        'reds': 113,
        'athletics': 133,
        'mariners': 136,
        'dodgers': 119,
}
FULL_TEAM_NAMES = {
        'texas rangers': 'rangers',
        'los angeles angels': 'angels',
        'houston astros': 'astros',
        'tampa bay rays': 'rays',
        'toronto blue jays': 'blue_jays',
        'new york yankees': 'yankees',
        'baltimore orioles': 'orioles',
        'boston red sox': 'red_sox',
        'minnesota twins': 'twins',
        'chicago white sox': 'white_sox',
        'cleveland guardians': 'guardians',
        'detroit tigers': 'tigers',
        'kansas city royals': 'royals',
        'san diego padres': 'padres',
        'san francisco giants': 'giants',
        'arizona diamondbacks': 'diamondbacks',
        'colorado rockies': 'rockies',
        'philadelphia phillies': 'phillies',
        'atlanta braves': 'braves',
        'miami marlins': 'marlins',
        'washington nationals': 'nationals',
        'new york mets': 'mets',
        'pittsburgh pirates': 'pirates',
        'st louis cardinals': 'cardinals',
        'milwaukee brewers': 'brewers',
        'chicago cubs': 'cubs',
        'cincinnati reds': 'reds',
        'oakland athletics': 'athletics',
        'seattle mariners': 'mariners',
        'los angeles dodgers': 'dodgers',
}


PROJECT_ID = "gem-rush-007"  # Replace with your actual project ID
LOCATION = "us-central1"  # Or your preferred region
DATASET_ID = "mlb_data_2024"
TABLE_ID = "game_events_hybrid"  # Modified table name for clarity
BUCKET_URI = f"gs://{PROJECT_ID}-vs-hybridsearch-mlb"
VECTORIZER_FILE = "tfidf_vectorizer.json.pkl"
# Initialize Vertex AI and BigQuery
aiplatform.init(project=PROJECT_ID, location=LOCATION)
bq_client = bigquery.Client(project=PROJECT_ID)
embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-005")

# BigQuery client setup
client = bigquery.Client()
CALLS = 100
RATE_LIMIT = 60

# --- BigQuery Table Schema ---
# Add sparse_embedding to the schema

GAME_EVENTS_SCHEMA = [
    bigquery.SchemaField("game_id", "INTEGER", mode="REQUIRED"),
    bigquery.SchemaField("play_id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("official_date", "DATE", mode="REQUIRED"),
    bigquery.SchemaField("game_type", "STRING"),
    bigquery.SchemaField("season", "INTEGER"),
    bigquery.SchemaField("home_team_name", "STRING"),
    bigquery.SchemaField("away_team_name", "STRING"),
    bigquery.SchemaField("venue_name", "STRING"),
    bigquery.SchemaField("inning", "INTEGER"),
    bigquery.SchemaField("half_inning", "STRING"),
    bigquery.SchemaField("event", "STRING"),
    bigquery.SchemaField("event_type", "STRING"),
    bigquery.SchemaField("description", "STRING"),
    bigquery.SchemaField("batter_name", "STRING"),
    bigquery.SchemaField("pitcher_name", "STRING"),
    bigquery.SchemaField("rich_text", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("embedding", "FLOAT64", mode="REPEATED"),  # Dense embedding
    bigquery.SchemaField("sparse_embedding_values", "FLOAT64", mode="REPEATED"), # Sparse embedding values
    bigquery.SchemaField("sparse_embedding_dimensions", "INTEGER", mode="REPEATED"), # Sparse embedding dimensions
    bigquery.SchemaField("last_updated", "TIMESTAMP")
]

def create_dataset(dataset_id=DATASET_ID):
    """Creates a BigQuery dataset if it doesn't exist."""
    try:
        dataset = bigquery.Dataset(f"{PROJECT_ID}.{dataset_id}")
        dataset.location = "US"
        client.create_dataset(dataset, exists_ok=True)
        logger.info(f"Dataset {dataset_id} created or already exists")
    except Exception as e:
        logger.error(f"Error creating dataset: {str(e)}")
        raise

def create_storage_bucket(bucket_name=None):
    """Creates a Cloud Storage bucket programmatically."""
    if bucket_name is None:
        bucket_name = BUCKET_URI.replace("gs://", "")
    try:
        storage_client = storage.Client(project=PROJECT_ID)
        if storage_client.lookup_bucket(bucket_name):
            logger.info(f"Bucket {bucket_name} already exists")
            return True
        bucket = storage_client.create_bucket(bucket_name, location="us-central1")
        logger.info(f"Bucket {bucket.name} created successfully")
        return True
    except Exception as e:
        logger.error(f"Error creating bucket {bucket_name}: {str(e)}")
        return False

def create_game_events_table(dataset_id: str = DATASET_ID, table_id: str = TABLE_ID):
    """Creates the unified BigQuery table."""
    dataset_ref = client.dataset(dataset_id)
    table_ref = dataset_ref.table(table_id)
    table = bigquery.Table(table_ref, schema=GAME_EVENTS_SCHEMA)
    try:
        client.create_table(table)
        logger.info(f"Created {table_id} table")
    except Exception as e:
        logger.info(f"{table_id} table already exists: {str(e)}")

def get_dense_embedding(text: str) -> List[float]:
    """Gets the dense embedding using Vertex AI."""
    embedding = embedding_model.get_embeddings([text])[0].values
    return embedding

def get_sparse_embedding(text: str, vectorizer) -> Tuple[List[float], List[int]]:
    """Gets the sparse embedding using the trained TF-IDF vectorizer."""
    tfidf_vector = vectorizer.transform([text])
    values = []
    dims = []
    for i, tfidf_value in enumerate(tfidf_vector.data):
        values.append(float(tfidf_value))
        dims.append(int(tfidf_vector.indices[i]))
    return values, dims

def generate_embeddings_and_upload(game_events: List[Dict], vectorizer):
    """Generates both dense and sparse embeddings and uploads to BigQuery."""
    if not game_events:
        return
    df = pd.DataFrame(game_events)
    # Check if official_date needs conversion
    if pd.api.types.is_datetime64_any_dtype(df['official_date']):
        df['official_date'] = df['official_date'].dt.date
    # If it's already a date object, no conversion needed
    df['embedding'] = df["rich_text"].apply(get_dense_embedding)
    sparse_embeddings = df["rich_text"].apply(lambda x: get_sparse_embedding(x, vectorizer))
    df['sparse_embedding_values'] = sparse_embeddings.apply(lambda x: x[0])
    df['sparse_embedding_dimensions'] = sparse_embeddings.apply(lambda x: x[1])
    job_config = bigquery.LoadJobConfig(schema=GAME_EVENTS_SCHEMA, write_disposition="WRITE_APPEND")
    job = bq_client.load_table_from_dataframe(df, f"{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}", job_config=job_config)
    try:
        job.result()
        logger.info(f"Uploaded {len(df)} rows to BigQuery")
    except Exception as e:
        logger.error(f"Error uploading to BigQuery: {str(e)}")
        if job.errors:
            for error in job.errors:
                logger.error(f"Detailed error: {error}")

def get_team_key(team_name: str) -> Optional[str]:
    """Retrieves the team key (short name) from a team name."""
    team_name = team_name.lower().strip()
    if team_name in TEAMS:
        return team_name
    if team_name in FULL_TEAM_NAMES:
        return FULL_TEAM_NAMES[team_name]
    for full_name, short_name in FULL_TEAM_NAMES.items():
        if team_name in full_name:
            return short_name
    for short_name in TEAMS:
        if team_name in short_name:
            return short_name
    return None

def _get_table_name(team_name: str) -> str:
    """Helper function to construct the source table name."""
    team_key = get_team_key(team_name)
    if team_key:
        return f"`gem-rush-007.{team_key}_mlb_data_2024`"  # 2024 source tables
    return None

def fetch_team_data_from_bq(team_name: str, limit: int = 100) -> List[Dict]:
    """Fetches game data from existing BQ tables (2024 data)."""
    table_name = _get_table_name(team_name)
    if not table_name:
        logger.error(f"Could not determine table name for team: {team_name}")
        return []

    query = f"""
    SELECT
        g.game_id,
        g.official_date,
        g.game_type,
        g.season,
        g.home_team_name,
        g.away_team_name,
        g.venue_name,
        p.inning,
        p.half_inning,
        p.event,
        p.event_type,
        p.description,
        r_batter.full_name as batter_name,
        r_pitcher.full_name as pitcher_name
    FROM
        {table_name}.games AS g
    INNER JOIN
        {table_name}.plays AS p
    ON
        g.game_id = p.game_id
    LEFT JOIN
        {table_name}.roster AS r_batter
    ON
        p.batter_id = r_batter.player_id
    LEFT JOIN
        {table_name}.roster AS r_pitcher
    ON
        p.pitcher_id = r_pitcher.player_id
    ORDER BY
        g.official_date DESC, p.end_time DESC
    LIMIT @limit
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("limit", "INT64", limit),
        ]
    )
    try:
        query_job = bq_client.query(query, job_config=job_config)
        results = list(query_job.result())
        return [dict(row) for row in results]
    except Exception as e:
        logger.error(f"Error fetching data from BigQuery for {team_name}: {e}")
        return []

def process_bq_data(game_data: List[Dict]) -> List[Dict]:
    """Processes data fetched from BQ, creating 'rich_text'."""
    processed_events = []
    for row in game_data:

        # Ensure official_date is a date object
        if isinstance(row['official_date'], datetime):
            date_str = row['official_date'].strftime('%Y-%m-%d')
        else:
            # If it's already a date object or string, just use it as is
            date_str = str(row['official_date'])

        rich_text = (
            f"On {row['official_date']}, in a {row['game_type']} game of the {row['season']} season, "
            f"the {row['home_team_name']} played against the {row['away_team_name']} at {row['venue_name']}. "
            f"In the {row['half_inning']} of the {row['inning']} inning, "
            f"{row['batter_name']} (batter) faced {row['pitcher_name']} (pitcher). "
            f"The play resulted in a {row['event']} ({row['event_type']}). "
            f"Description: {row['description']}."
        )
        event_data = {
            'game_id': row['game_id'],
            'play_id': f"{row['game_id']}_{row.get('play_id', 'UNKNOWN')}",  # Handle potential missing play_id
            'official_date': row['official_date'],
            'game_type': row['game_type'],
            'season': row['season'],
            'home_team_name': row['home_team_name'],
            'away_team_name': row['away_team_name'],
            'venue_name': row['venue_name'],
            'inning': row['inning'],
            'half_inning': row['half_inning'],
            'event': row['event'],
            'event_type': row['event_type'],
            'description': row['description'],
            'batter_name': row['batter_name'],
            'pitcher_name': row['pitcher_name'],
            'rich_text': rich_text,
            'embedding': [],
            'sparse_embedding_values': [],
            'sparse_embedding_dimensions': [],
            'last_updated': datetime.now(UTC)
        }
        processed_events.append(event_data)
    return processed_events
def update_mlb_data_from_bq():
    """Fetches data from existing BQ tables, processes, generates embeddings."""
    logger.info("Starting MLB data update process from BigQuery...")
    create_dataset()
    create_game_events_table()

    # Train TF-IDF vectorizer (as before, but using potentially more data)
    all_rich_texts = []
    for team_name in TEAMS:  # Iterate through team *names*
        team_data = fetch_team_data_from_bq(team_name)  # Fetch from BQ!
        if team_data:
            processed_team_data = process_bq_data(team_data)
            all_rich_texts.extend([event['rich_text'] for event in processed_team_data])
    vectorizer = TfidfVectorizer()
    if all_rich_texts:
        vectorizer.fit(all_rich_texts)
        logger.info(f"Vectorizer fitted with {len(all_rich_texts)} existing text samples.")
    else:
        dummy_texts = ["baseball game home run", "pitcher strikeout inning", "batter hit double", "team win stadium","MLB baseball player"]
        vectorizer.fit(dummy_texts)
        logger.info("Vectorizer fitted with dummy data (no existing data found).")

    # Save the vectorizer after fitting
    bucket_name = BUCKET_URI.replace("gs://", "")
    save_success = save_vectorizer(vectorizer, bucket_name)
    if save_success:
        logger.info("Vectorizer saved successfully.")
    else:
        logger.warning("Failed to save vectorizer.")
        
    for team_name in TEAMS:
        team_data = fetch_team_data_from_bq(team_name)
        if not team_data:
            logger.info(f"No data found for team {team_name} in BigQuery.")
            continue
        logger.info(f"Processing data for team: {team_name}")
        processed_team_data = process_bq_data(team_data)
        if processed_team_data:
           generate_embeddings_and_upload(processed_team_data, vectorizer)
        else:
           logger.warning(f"No data to upload for team: {team_name}")

    logger.info("MLB data update process from BigQuery completed.")

def upload_file_to_bucket(source_file_path, bucket_name, destination_blob_name=None):
    """Uploads a file to a Google Cloud Storage bucket."""
    if destination_blob_name is None:
        destination_blob_name = os.path.basename(source_file_path)
    if bucket_name.startswith("gs://"):
        bucket_name = bucket_name.replace("gs://", "")
    try:
        storage_client = storage.Client(project=PROJECT_ID)
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(source_file_path)
        logger.info(f"File {source_file_path} uploaded to {bucket_name}/{destination_blob_name}")
        return True
    except Exception as e:
        logger.error(f"Error uploading file to bucket: {str(e)}")
        return False

# --- Vertex AI Vector Search Index Creation and Querying ---

def create_vector_search_index(bucket_uri: str = BUCKET_URI):
    """Creates a Vertex AI Vector Search index (both dense and sparse)."""
    jsonl_data = []
    query = f"SELECT game_id, play_id, embedding, sparse_embedding_values, sparse_embedding_dimensions FROM `{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}`"
    query_job = bq_client.query(query)
    results = query_job.result()

    for row in results:
      if row.embedding and row.sparse_embedding_values and row.sparse_embedding_dimensions:
        item_id = f"{row.game_id}_{row.play_id}"
        jsonl_item = {
            "id": item_id,
            "embedding": row.embedding,
            "sparse_embedding": {
                "values": row.sparse_embedding_values,
                "dimensions": row.sparse_embedding_dimensions
            }
        }
        jsonl_data.append(jsonl_item)
    # Use proper JSON format instead of string representations
    import json
    json_file_path = "mlb_game_events.json"
    with open(json_file_path, "w") as f:
        for item in jsonl_data:
            # Use json.dumps to properly format each JSON object
            f.write(json.dumps(item) + "\n")
    bucket_name = BUCKET_URI.replace("gs://", "")
    upload_success = upload_file_to_bucket(json_file_path, bucket_name)

    if not upload_success:
       logger.error("Failed to upload JSONL file to Cloud Storage")
       return None
    index_data_uri = f"{BUCKET_URI}/{json_file_path}"  # Point to the *specific* file
    try:
        my_index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
            display_name=f"mlb-hybrid-index-{datetime.now().strftime('%m%d%H%M')}",
            contents_delta_uri=index_data_uri,
            dimensions=768,
            approximate_neighbors_count=10,
        )
        logger.info(f'Vertex AI index created: {my_index.display_name}')
        return my_index

    except Exception as e:
        logger.error(f"Error creating index: {e}")
        return None
def save_vectorizer(vectorizer, bucket_name, filename=VECTORIZER_FILE):
    """Saves the fitted vectorizer to a GCS bucket."""
    try:
        with open(filename, 'wb') as f:
            pickle.dump(vectorizer, f)
        if bucket_name.startswith("gs://"):
            bucket_name = bucket_name.replace("gs://", "")
        upload_success = upload_file_to_bucket(filename, bucket_name, filename)
        if upload_success:
            logger.info(f"Vectorizer saved to {bucket_name}/{filename}")
            return True
        else:
            logger.error(f"Failed to upload vectorizer to {bucket_name}/{filename}")
            return False
    except Exception as e:
        logger.error(f"Error saving vectorizer: {e}")
        return False

def load_vectorizer(bucket_name, filename=VECTORIZER_FILE):
    """Loads the fitted vectorizer from a GCS bucket."""
    try:
        if bucket_name.startswith("gs://"):
            bucket_name = bucket_name.replace("gs://", "")
        storage_client = storage.Client(project=PROJECT_ID)
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(filename)
        if not blob.exists():
            logger.info(f"No saved vectorizer found at {bucket_name}/{filename}")
            return None
        blob.download_to_filename(filename)
        with open(filename, 'rb') as f:
            vectorizer = pickle.load(f)
        logger.info(f"Vectorizer loaded from {bucket_name}/{filename}")
        return vectorizer
    except Exception as e:
        logger.error(f"Error loading vectorizer: {e}")
        return None

def deploy_index(my_index):
      """Deploys the Vector Search index to an endpoint."""
      try:
          my_index_endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
              display_name=f"mlb-hybrid-index-endpoint-{datetime.now().strftime('%m%d%H%M')}",
              public_endpoint_enabled=True
          )
          DEPLOYED_INDEX_ID = f"mlb_hybrid_deployed_{datetime.now().strftime('%m%d%H%M')}"
          my_index_endpoint.deploy_index(
              index=my_index, deployed_index_id=DEPLOYED_INDEX_ID
          )
          logger.info(f"Index deployed to endpoint: {my_index_endpoint.display_name}")
          return my_index_endpoint, DEPLOYED_INDEX_ID

      except Exception as e:
        logger.error(f"Error deploying index: {e}")
        return None, None

def query_vector_search(
    index_endpoint, deployed_index_id: str, query_text: str, vectorizer, num_neighbors: int = 5
):
    """Queries the Vector Search index with a hybrid query."""
    from google.cloud.aiplatform.matching_engine.matching_engine_index_endpoint import (
    HybridQuery,
)
    query_dense_emb = get_dense_embedding(query_text)
    query_sparse_emb_values, query_sparse_emb_dimensions = get_sparse_embedding(query_text, vectorizer)
    query = HybridQuery(
      dense_embedding=query_dense_emb,
      sparse_embedding_dimensions=query_sparse_emb_dimensions,
      sparse_embedding_values=query_sparse_emb_values,
      rrf_ranking_alpha=0.5,
    )
    try:
        response = index_endpoint.find_neighbors(
            deployed_index_id=deployed_index_id,
            queries=[query],
            num_neighbors=num_neighbors,
        )
        if not response or len(response) == 0:
            logger.warning("Vector search returned empty response")
            return []
        results = []
        for neighbor in response[0]:
          query = f"""
              SELECT game_id, play_id, official_date, home_team_name, away_team_name, venue_name,
                     inning, half_inning, event, batter_name, pitcher_name, rich_text
              FROM `{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}`
              WHERE CONCAT(CAST(game_id as STRING),'_', play_id) = '{neighbor.id}'
          """
          query_job = bq_client.query(query)
          bq_results = list(query_job.result())
          if bq_results:
            result_dict = {
                "game_id": bq_results[0].game_id,
                "play_id": bq_results[0].play_id,
                "official_date": bq_results[0].official_date,
                "home_team_name": bq_results[0].home_team_name,
                "away_team_name": bq_results[0].away_team_name,
                "venue_name": bq_results[0].venue_name,
                "inning": bq_results[0].inning,
                "half_inning": bq_results[0].half_inning,
                "event": bq_results[0].event,
                "batter_name": bq_results[0].batter_name,
                "pitcher_name": bq_results[0].pitcher_name,
                "rich_text": bq_results[0].rich_text,
                "dense_distance": neighbor.distance if neighbor.distance else 0.0,
                "sparse_distance": neighbor.sparse_distance if neighbor.sparse_distance else 0.0,
              }
            results.append(result_dict)
        return results
    except Exception as e:
      logger.error(f"Error querying Vector Search: {e}")
      return []

def delete_bucket(bucket_name):
    """Deletes a Google Cloud Storage bucket and all its contents."""
    if bucket_name.startswith("gs://"):
        bucket_name = bucket_name.replace("gs://", "")
    try:
        storage_client = storage.Client(project=PROJECT_ID)
        bucket = storage_client.bucket(bucket_name)
        blobs = list(bucket.list_blobs())
        for blob in blobs:
            blob.delete()
        bucket.delete()
        logger.info(f"Bucket {bucket_name} deleted successfully")
        return True
    except Exception as e:
        logger.error(f"Error deleting bucket {bucket_name}: {str(e)}")
        return False

def clean_up(index_endpoint, my_index, bucket_uri):
  if index_endpoint:
    index_endpoint.undeploy_all()
    index_endpoint.delete(force=True)
  if my_index:
    my_index.delete()
  bucket_name = bucket_uri.replace("gs://", "")
  delete_success = delete_bucket(bucket_name)
  if not delete_success:
    logger.error(f"Error deleting Cloud Storage bucket: {bucket_name}")
  else:
    logger.info(f"Successfully deleted Cloud Storage bucket: {bucket_name}")

if __name__ == "__main__":
    create_dataset()
    bucket_name = BUCKET_URI.replace("gs://", "")
    bucket_created = create_storage_bucket()
    if not bucket_created:
        logger.error("Failed to create bucket, exiting")
        sys.exit(1)
    update_mlb_data_from_bq()  # Use the new BQ-based ingestion
    my_index = create_vector_search_index()
    if my_index:
        index_endpoint, deployed_index_id = deploy_index(my_index)
        vectorizer = load_vectorizer(bucket_name)

        if index_endpoint and deployed_index_id and vectorizer:
            query1 = "home runs by the Rangers against the Astros in the 9th inning"
            results1 = query_vector_search(index_endpoint, deployed_index_id, query1, vectorizer)
            print(f"\nResults for query '{query1}':")
            for result in results1:
                print(result)

            query2 = "strikeouts at Yankee Stadium"
            results2 = query_vector_search(index_endpoint, deployed_index_id, query2, vectorizer)
            print(f"\nResults for query '{query2}':")
            for result in results2:
                print(result)

            query3 = "games on 2024-05-15"
            results3 = query_vector_search(index_endpoint, deployed_index_id, query3, vectorizer)
            print(f"\nResults for query '{query3}':")
            for result in results3:
                print(result)

            query4 = "a goal scored by Messi"  # Irrelevant to MLB
            results4 = query_vector_search(index_endpoint, deployed_index_id, query4, vectorizer)
            print(f"\nResults for query '{query4}':")  # Should be empty.
            for result in results4:
              print(result)
    # clean_up(index_endpoint, my_index, BUCKET_URI)