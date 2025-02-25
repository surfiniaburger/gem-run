
from datetime import datetime, date, UTC, timedelta
import logging
import google.cloud.logging
from google.cloud.logging.handlers import CloudLoggingHandler
from google.cloud import secretmanager  # You might not need this if not using secrets
# from pymongo.mongo_client import MongoClient # No MongoDB
# from pymongo.server_api import ServerApi # No MongoDB
import pandas as pd
from google.cloud import bigquery
from typing import Dict, List, Tuple
from vertexai.language_models import TextEmbeddingModel  
from google.cloud import aiplatform  # Use the Vertex AI SDK
import numpy as np
# from pymongo.operations import SearchIndexModel # No MongoDB
import requests
from ratelimit import limits, sleep_and_retry
from sklearn.feature_extraction.text import TfidfVectorizer # For sparse embeddings
import sys
import os
from google.cloud import storage
import pickle  # For saving/loading the vectorizer

# --- Setup (Logging, BigQuery Client, Rate Limiting) ---

def setup_logging():
    """Sets up Google Cloud Logging."""
    client = google.cloud.logging.Client()
    handler = CloudLoggingHandler(client)
    logger = logging.getLogger('bigquery_vector_search_hybrid') # Changed logger name
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
BUCKET_URI = f"gs://{PROJECT_ID}-vs-hybridsearch-mlb" # Example bucket URI.  Create this bucket!
VECTORIZER_FILE = "tfidf_vectorizer.pkl"  # File to store the fitted vectorizer
# Initialize Vertex AI and BigQuery
aiplatform.init(project=PROJECT_ID, location=LOCATION)
bq_client = bigquery.Client(project=PROJECT_ID)
embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-005")

# BigQuery client setup
client = bigquery.Client()
CALLS = 100
RATE_LIMIT = 60

@sleep_and_retry
@limits(calls=CALLS, period=RATE_LIMIT)
def call_mlb_api(url: str) -> Dict:
    """Make a rate-limited call to the MLB API."""
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

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
        dataset.location = "US"  # Specify the location matching your project
        client.create_dataset(dataset, exists_ok=True)
        logger.info(f"Dataset {dataset_id} created or already exists")
    except Exception as e:
        logger.error(f"Error creating dataset: {str(e)}")
        raise

def create_storage_bucket(bucket_name=None):
    """Creates a Cloud Storage bucket programmatically."""
    if bucket_name is None:
        # Extract bucket name from the bucket URI
        bucket_name = BUCKET_URI.replace("gs://", "")
    
    try:
        # Initialize the storage client
        storage_client = storage.Client(project=PROJECT_ID)
        
        # Check if bucket already exists
        if storage_client.lookup_bucket(bucket_name):
            logger.info(f"Bucket {bucket_name} already exists")
            return True
            
        # Create a new bucket
        bucket = storage_client.create_bucket(bucket_name, location="us-central1")  # You can change the location
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

# --- Data Retrieval and Processing ---
def get_roster_data(game_pk: int) -> Dict[int, str]:
    """Fetches the roster for a game."""
    url = f'https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live'
    game_data = call_mlb_api(url)
    players = game_data['liveData']['boxscore']['teams']
    roster = {}
    for home_away in ['home', 'away']:
        for player in players[home_away]['players'].values():
           roster[player['person']['id']] = player['person']['fullName']
    return roster

def get_team_games(team_id: int, season: int = 2024, num_games: int = 2) -> List[Dict]:
    """Fetch recent games for a team."""
    url = f'https://statsapi.mlb.com/api/v1/schedule?sportId=1&season={season}&teamId={team_id}&gameType=R'
    schedule_data = call_mlb_api(url)
    games_list = []
    for date_entry in schedule_data.get('dates', []):
        for game in date_entry.get('games', []):
            games_list.append({
                'game_id': game['gamePk'],
                'official_date': game['officialDate'],
                'game_type': game['gameType'],
                'season': season,
                'home_team_name': game['teams']['home']['team']['name'],
                'away_team_name': game['teams']['away']['team']['name'],
                'venue_name': game['venue']['name'],
                'status': game['status']['detailedState']
            })
    games_list = sorted(games_list, key=lambda x: x['official_date'], reverse=True)[:num_games]
    return games_list


def process_game_data(game_info: Dict, roster: Dict[int, str]) -> List[Dict]:
    """Fetches and processes play-by-play data."""
    game_pk = game_info['game_id']
    url = f'https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live'
    game_data = call_mlb_api(url)
    all_plays = game_data['liveData']['plays']['allPlays']
    game_events = []

    for play in all_plays:
        if not all([play.get('about'), play.get('matchup'), play.get('result'), play.get('count')]):
            logger.warning(f"Skipping play in game {game_pk} due to missing data: {play}")
            continue

        play_id = f"{game_pk}_{play['about']['atBatIndex']}"
        batter_id = play['matchup']['batter']['id']
        pitcher_id = play['matchup']['pitcher']['id']
        batter_name = roster.get(batter_id, "Unknown Batter")
        pitcher_name = roster.get(pitcher_id, "Unknown Pitcher")
        rich_text = (
            f"On {game_info['official_date']}, in a {game_info['game_type']} game of the {game_info['season']} season, "
            f"the {game_info['home_team_name']} played against the {game_info['away_team_name']} at {game_info['venue_name']}. "
            f"In the {play['about']['halfInning']} of the {play['about']['inning']} inning, "
            f"{batter_name} (batter) faced {pitcher_name} (pitcher). "
            f"The play resulted in a {play['result']['event']} ({play['result']['eventType']}). "
            f"Description: {play['result']['description']}."
        )
        event_data = {
            'game_id': game_pk,
            'play_id': play_id,
            'official_date': game_info['official_date'],
            'game_type': game_info['game_type'],
            'season': game_info['season'],
            'home_team_name': game_info['home_team_name'],
            'away_team_name': game_info['away_team_name'],
            'venue_name': game_info['venue_name'],
            'inning': play['about']['inning'],
            'half_inning': play['about']['halfInning'],
            'event': play['result']['event'],
            'event_type': play['result']['eventType'],
            'description': play['result']['description'],
            'batter_name': batter_name,
            'pitcher_name': pitcher_name,
            'rich_text': rich_text,
            'embedding': [],  # Placeholder for dense embedding
            'sparse_embedding_values': [], # Placeholder for sparse embedding values
            'sparse_embedding_dimensions': [], # Placeholder for sparse embedding dimensions
            'last_updated': datetime.now(UTC)
        }
        game_events.append(event_data)

    return game_events

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

    # Convert 'official_date' from string to pandas datetime type
    # This ensures proper conversion to BigQuery DATE type
    df = pd.DataFrame(game_events)
    df['official_date'] = df['official_date'].dt.date

    # Generate dense embeddings
    df['embedding'] = df["rich_text"].apply(get_dense_embedding)

    # Generate sparse embeddings
    sparse_embeddings = df["rich_text"].apply(lambda x: get_sparse_embedding(x, vectorizer))
    df['sparse_embedding_values'] = sparse_embeddings.apply(lambda x: x[0])
    df['sparse_embedding_dimensions'] = sparse_embeddings.apply(lambda x: x[1])

    # Upload to BigQuery
    job_config = bigquery.LoadJobConfig(schema=GAME_EVENTS_SCHEMA, write_disposition="WRITE_APPEND")
    
    # Convert datetime.date objects to strings in the format BigQuery expects
    job = bq_client.load_table_from_dataframe(df, f"{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}", job_config=job_config)
    try:
        job.result()  # Wait for the job to complete
        logger.info(f"Uploaded {len(df)} rows to BigQuery")
    except Exception as e:
        logger.error(f"Error uploading to BigQuery: {str(e)}")
        # Print more detailed error information if available
        if job.errors:
            for error in job.errors:
                logger.error(f"Detailed error: {error}")

def update_mlb_data():
    """Fetches new game data, processes, generates embeddings, and updates BigQuery."""
    logger.info("Starting MLB data update process...")
    # Create dataset first
    create_dataset()

    create_game_events_table()

    # Train TF-IDF vectorizer ONCE, using ALL existing data
    all_rich_texts = []
    for team_id in TEAMS.values():
        try:
            recent_games = get_team_games(team_id)
            for game_info in recent_games:
                game_pk = game_info['game_id']
                query = f"SELECT rich_text FROM `{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}` WHERE game_id = {game_pk}"
                query_job = bq_client.query(query)
                results = list(query_job.result())
                all_rich_texts.extend([row.rich_text for row in results]) #getting all the rich_texts
        except Exception as e:
            logger.error(f"Error getting existing texts for team ID {team_id}: {e}")
            continue
    # Fit the Vectorizer - Make sure it's always fitted even with no data
    vectorizer = TfidfVectorizer()
    
    if all_rich_texts:
        # If we have existing data, fit with that
        try:
            vectorizer.fit(all_rich_texts)
            logger.info(f"Vectorizer fitted with {len(all_rich_texts)} existing text samples.")
        except Exception as e:
            logger.error(f"Vectorizer fitting failed with existing data: {e}")
            # Still need to fit with something, so we'll use dummy data below
            all_rich_texts = []
    
    # If no existing data or fitting failed, fit with dummy data
    if not all_rich_texts:
        dummy_texts = [
            "baseball game home run", 
            "pitcher strikeout inning", 
            "batter hit double", 
            "team win stadium",
            "MLB baseball player"
        ]
        try:
            vectorizer.fit(dummy_texts)
            logger.info("Vectorizer fitted with dummy data as no existing rich_text was available.")
        except Exception as e:
            logger.error(f"Vectorizer fitting with dummy data failed: {e}")
            return  # Cannot proceed without a fitted vectorizer


    for team_id in TEAMS.values():
        try:
            recent_games = get_team_games(team_id)
            for game_info in recent_games:
                game_pk = game_info['game_id']
                query = f"SELECT game_id FROM `{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}` WHERE game_id = {game_pk} LIMIT 1"
                query_job = bq_client.query(query)
                results = list(query_job.result())

                if not results:
                    logger.info(f"Processing new game: {game_pk}")
                    roster = get_roster_data(game_pk)
                    game_events = process_game_data(game_info, roster)
                    generate_embeddings_and_upload(game_events, vectorizer)  # Pass vectorizer
                else:
                    logger.info(f"Game {game_pk} already exists in BigQuery. Skipping.")

        except Exception as e:
            logger.error(f"Error updating data for team ID {team_id}: {e}")
            continue

    logger.info("MLB data update process completed.")

def upload_file_to_bucket(source_file_path, bucket_name, destination_blob_name=None):
    """Uploads a file to a Google Cloud Storage bucket."""
    if destination_blob_name is None:
        destination_blob_name = os.path.basename(source_file_path)
    
    if bucket_name.startswith("gs://"):
        bucket_name = bucket_name.replace("gs://", "")
        
    try:
        # Initialize the storage client
        storage_client = storage.Client(project=PROJECT_ID)
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        
        # Upload the file
        blob.upload_from_filename(source_file_path)
        logger.info(f"File {source_file_path} uploaded to {bucket_name}/{destination_blob_name}")
        return True
    except Exception as e:
        logger.error(f"Error uploading file to bucket: {str(e)}")
        return False

# --- Vertex AI Vector Search Index Creation and Querying ---

def create_vector_search_index(bucket_uri: str = BUCKET_URI):
    """Creates a Vertex AI Vector Search index (both dense and sparse)."""
    # Prepare data for Vector Search (create JSONL file)
    jsonl_data = []
    query = f"SELECT game_id, play_id, embedding, sparse_embedding_values, sparse_embedding_dimensions FROM `{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}`"
    query_job = bq_client.query(query)
    results = query_job.result()

    for row in results:
      #Make sure to skip if any of the embeddings don't exist
      if row.embedding and row.sparse_embedding_values and row.sparse_embedding_dimensions:
        item_id = f"{row.game_id}_{row.play_id}"  # Unique ID
        jsonl_item = {
            "id": item_id,
            "embedding": row.embedding,  # Dense embedding
            "sparse_embedding": {
                "values": row.sparse_embedding_values,
                "dimensions": row.sparse_embedding_dimensions
            }
        }
        jsonl_data.append(jsonl_item)

    # Write to a JSONL file
    jsonl_file_path = "mlb_game_events.jsonl"  # Local file
    with open(jsonl_file_path, "w") as f:
        for item in jsonl_data:
            f.write(f"{item}\n")



    # Replace the gsutil command with:
    bucket_name = BUCKET_URI.replace("gs://", "")
    upload_success = upload_file_to_bucket(jsonl_file_path, bucket_name)
    if not upload_success:
       logger.error("Failed to upload JSONL file to Cloud Storage")
       return None

    # Create the index
    try:
        my_index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
            display_name=f"mlb-hybrid-index-{datetime.now().strftime('%m%d%H%M')}",
            contents_delta_uri=bucket_uri,
            dimensions=768,  # Dense embedding dimensions
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
        # Save locally first
        with open(filename, 'wb') as f:
            pickle.dump(vectorizer, f)
        
        # Upload to GCS
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
        
        # Download from GCS to local file
        storage_client = storage.Client(project=PROJECT_ID)
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(filename)
        
        if not blob.exists():
            logger.info(f"No saved vectorizer found at {bucket_name}/{filename}")
            return None
        
        blob.download_to_filename(filename)
        
        # Load the vectorizer
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
          # Create IndexEndpoint
          my_index_endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
              display_name=f"mlb-hybrid-index-endpoint-{datetime.now().strftime('%m%d%H%M')}",
              public_endpoint_enabled=True
          )

          # Deploy the index
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
    # Get dense embedding
    query_dense_emb = get_dense_embedding(query_text)

    # Get sparse embedding
    query_sparse_emb_values, query_sparse_emb_dimensions = get_sparse_embedding(query_text, vectorizer)

    # Create HybridQuery
    query = HybridQuery(
      dense_embedding=query_dense_emb,
      sparse_embedding_dimensions=query_sparse_emb_dimensions,
      sparse_embedding_values=query_sparse_emb_values,
      rrf_ranking_alpha=0.5,  # Adjust as needed
    )

    # Run the query
    try:
        response = index_endpoint.find_neighbors(
            deployed_index_id=deployed_index_id,
            queries=[query],
            num_neighbors=num_neighbors,
        )

        # Process results
        results = []
        for neighbor in response[0]:
          # Fetch details from BigQuery.  This is important!
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
        
        # Delete all blobs in the bucket
        blobs = list(bucket.list_blobs())
        for blob in blobs:
            blob.delete()
            
        # Delete the bucket
        bucket.delete()
        logger.info(f"Bucket {bucket_name} deleted successfully")
        return True
    except Exception as e:
        logger.error(f"Error deleting bucket {bucket_name}: {str(e)}")
        return False


def clean_up(index_endpoint, my_index, bucket_uri):

  # delete Index Endpoint
  if index_endpoint:
    index_endpoint.undeploy_all()
    index_endpoint.delete(force=True)

  # delete Indexes
  if my_index:
    my_index.delete()

  # delete Cloud Storage bucket
  bucket_name = bucket_uri.replace("gs://", "")
  delete_success = delete_bucket(bucket_name)
  if not delete_success:
    logger.error(f"Error deleting Cloud Storage bucket: {bucket_name}")
  else:
    logger.info(f"Successfully deleted Cloud Storage bucket: {bucket_name}")

if __name__ == "__main__":
        # Create dataset first
    create_dataset()
        
    # Create bucket
    bucket_name = BUCKET_URI.replace("gs://", "")    
    bucket_created = create_storage_bucket()
    if not bucket_created:
        logger.error("Failed to create bucket, exiting")
        sys.exit(1)        # Create storage bucket
    #create_storage_bucket()    
    # --- Initial setup (Run once) ---
    update_mlb_data()  # Fetch initial data and create tables
   
    # --- Create and deploy Vector Search index ---
    my_index = create_vector_search_index()
    
    if my_index:  # Proceed only if index creation was successful
        index_endpoint, deployed_index_id = deploy_index(my_index)
        
        # Try to load previously saved vectorizer
        vectorizer = load_vectorizer(bucket_name)
        
        # If no saved vectorizer, create and fit a new one
        if vectorizer is None:
            all_rich_texts = []
            for team_id in TEAMS.values():
                try:
                    recent_games = get_team_games(team_id)
                    for game_info in recent_games:
                        game_pk = game_info['game_id']
                        query = f"SELECT rich_text FROM `{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}` WHERE game_id = {game_pk}"
                        query_job = bq_client.query(query)
                        results = list(query_job.result())
                        all_rich_texts.extend([row.rich_text for row in results])
                except Exception as e:
                    logger.error(f"Error getting existing texts for team ID {team_id}: {e}")
                    continue
            
            vectorizer = TfidfVectorizer()
            if all_rich_texts:
                vectorizer.fit(all_rich_texts)
                logger.info(f"Vectorizer fitted with {len(all_rich_texts)} text samples")
            else:
                # Fallback to dummy data if no real data available
                dummy_texts = [
                    "baseball game home run", 
                    "pitcher strikeout inning", 
                    "batter hit double", 
                    "team win stadium",
                    "MLB baseball player"
                ]
                vectorizer.fit(dummy_texts)
                logger.info("Vectorizer fitted with dummy data")
            
            # Save the fitted vectorizer for future use
            save_vectorizer(vectorizer, bucket_name)

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
    #Clean Up
    # clean_up(index_endpoint, my_index, BUCKET_URI)