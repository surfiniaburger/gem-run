from datetime import datetime, date, UTC, timedelta
import logging
import google.cloud.logging
from google.cloud.logging.handlers import CloudLoggingHandler
from google.cloud import secretmanager
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import pandas as pd
from google.cloud import bigquery
from typing import Dict, List, Tuple
from vertexai.language_models import TextEmbeddingModel
import numpy as np
from pymongo.operations import SearchIndexModel
import requests
from ratelimit import limits, sleep_and_retry

# --- Setup (Logging, Secret Manager, BigQuery Client, Rate Limiting) ---

def setup_logging():
    """Sets up Google Cloud Logging (same as before)."""
    client = google.cloud.logging.Client()
    handler = CloudLoggingHandler(client)
    logger = logging.getLogger('mongodb_vector_search_unified')
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

def get_secret(project_id, secret_id, version_id="latest"):
    """Retrieves a secret from Google Cloud Secret Manager (same as before)."""
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"
    response = client.access_secret_version(request={"name": name})
    return response.payload.data.decode("UTF-8")

PROJECT_ID = "gem-rush-007"  # Replace with your actual project ID
bq_client = bigquery.Client(project=PROJECT_ID)
embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-005")

# BigQuery client setup
client = bigquery.Client()
CALLS = 100
RATE_LIMIT = 60

@sleep_and_retry
@limits(calls=CALLS, period=RATE_LIMIT)
def call_mlb_api(url: str) -> Dict:
    """Make a rate-limited call to the MLB API (same as before)."""
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

# --- MongoDB Connection and Database Functions ---

def connect_to_mongodb(project_id: str = "gem-rush-007", secret_id: str = "mongodb-uri"):
    """Connects to MongoDB using URI from Secret Manager (same as before)."""
    uri = get_secret(project_id, secret_id)
    client = MongoClient(uri, server_api=ServerApi('1'))
    client.admin.command('ping')
    logger.info("Successfully connected to MongoDB!")
    return client

def create_mongodb_database(client: MongoClient, db_name: str):
    """Creates a MongoDB database if it doesn't exist (same as before)."""
    db = client[db_name]
    if db_name not in client.list_database_names():
        db.dummy_collection.insert_one({"dummy": "data"})
        logger.info(f"MongoDB database '{db_name}' created.")
        db.dummy_collection.drop()
        logger.info("Dummy collection dropped.")
    else:
        logger.info(f"MongoDB database '{db_name}' already exists.")
    return db

# --- BigQuery Table Schema ---

GAME_EVENTS_SCHEMA = [
    bigquery.SchemaField("game_id", "INTEGER", mode="REQUIRED"),
    bigquery.SchemaField("play_id", "STRING", mode="REQUIRED"),  # Combined game_id and play index
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
    bigquery.SchemaField("description", "STRING"),  # Original MLB description
    bigquery.SchemaField("batter_name", "STRING"),
    bigquery.SchemaField("pitcher_name", "STRING"),
    bigquery.SchemaField("rich_text", "STRING", mode="REQUIRED"),  # The combined text for embedding
    bigquery.SchemaField("embedding", "FLOAT64", mode="REPEATED"), # The embedding vector
    bigquery.SchemaField("last_updated", "TIMESTAMP")
]

def create_game_events_table(dataset_id: str = "mlb_data_2024"):
    """Creates the unified BigQuery table."""
    dataset_ref = client.dataset(dataset_id)
    table_ref = dataset_ref.table("game_events")
    table = bigquery.Table(table_ref, schema=GAME_EVENTS_SCHEMA)
    try:
        client.create_table(table)
        logger.info("Created game_events table")
    except Exception as e:
        logger.info(f"game_events table already exists: {str(e)}")

# --- Data Retrieval and Processing ---
def get_roster_data(game_pk: int) -> Dict[int, str]:
    """Fetches the roster for a game and returns a player_id to name mapping."""
    url = f'https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live'
    game_data = call_mlb_api(url)
    players = game_data['liveData']['boxscore']['teams']
    roster = {}
    for home_away in ['home', 'away']:
        for player in players[home_away]['players'].values():
           roster[player['person']['id']] = player['person']['fullName']
    return roster

def get_team_games(team_id: int, season: int = 2024, num_games: int = 2) -> List[Dict]:
    """Fetch a specified number of recent games for a team."""
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

    # Sort games by date and get the last 'num_games'
    games_list = sorted(games_list, key=lambda x: x['official_date'], reverse=True)[:num_games]
    return games_list


def process_game_data(game_info: Dict, roster:Dict[int, str]) -> List[Dict]:
    """Fetches and processes play-by-play data for a single game."""
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
        # Construct the rich_text description
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
            'embedding': [],  # Placeholder, will be filled later
            'last_updated': datetime.now(UTC)
        }
        game_events.append(event_data)

    return game_events


def generate_embeddings_and_upload(game_events: List[Dict]):
    """Generates embeddings for the rich_text and uploads to BQ and MongoDB."""
    if not game_events:
        return

    df = pd.DataFrame(game_events)

     # Convert date columns to string before embedding
    df['official_date'] = df['official_date'].astype(str)

    # Generate embeddings
    try:
        embeddings = embedding_model.get_embeddings(df["rich_text"].tolist())
        df['embedding'] = [embedding.values for embedding in embeddings]
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        return
    
    # Explode embeddings for BigQuery
    df['embedding'] = df['embedding'].apply(lambda x: x if isinstance(x, list) else [])
    # Upload to BigQuery
    job_config = bigquery.LoadJobConfig(schema=GAME_EVENTS_SCHEMA, write_disposition="WRITE_APPEND")
    job = bq_client.load_table_from_dataframe(df, f"{PROJECT_ID}.mlb_data_2024.game_events", job_config=job_config)
    job.result()
    logger.info(f"Uploaded {len(df)} rows to BigQuery")

    # Prepare for and upload to MongoDB
    mongo_client = connect_to_mongodb()
    db = create_mongodb_database(mongo_client, "mlb_data")
    collection = db["game_events"]

    # Convert to list of dicts and insert into MongoDB
    records = df.to_dict("records")
    for record in records:
        if not record['embedding']:
            logger.warning(f"Skipping record due to empty embedding: {record}")
            continue
        try:
            collection.insert_one(record)
        except Exception as e:
            logger.error(f"Error inserting into MongoDB: {e}")
            continue

    logger.info(f"Inserted {len(records)} records into MongoDB")
    create_atlas_vector_search_index(mongo_client, "mlb_data", "game_events")  # Ensure index exists
    mongo_client.close()


def create_atlas_vector_search_index(client: MongoClient, db_name: str, collection_name: str):
    """Creates the Atlas Vector Search index (same as before, but adapted for clarity)."""
    db = client[db_name]
    collection = db[collection_name]

    try:
        existing_indexes = collection.list_search_indexes()
    except Exception as e:
        logger.warning("Could not list existing search indexes: %s", e)
        existing_indexes = []

    if any(idx.get('name') == "vector_index" for idx in existing_indexes):
        logger.info(f"Index 'vector_index' already exists on {db_name}.{collection_name}.")
        return

    index_definition = {
        "mappings": {
            "dynamic": False,
            "fields": {
                "embedding": {
                    "type": "knnVector",
                    "dimensions": 768,
                    "similarity": "dotProduct"
                }
            }
        }
    }
    search_index_model = SearchIndexModel(definition=index_definition, name="vector_index")
    try:
        collection.create_search_indexes([search_index_model])
        logger.info("Created Atlas Search index: vector_index")
    except Exception as e:
        logger.error(f"Error creating search index: {e}")
        raise

# --- Querying Function ---

def query_mongodb(query_text: str, limit: int = 5) -> List[Dict]:
    """Queries MongoDB Atlas using vector similarity."""
    mongo_client = connect_to_mongodb()
    db = mongo_client["mlb_data"]  # Use the correct database name
    collection = db["game_events"] # Use the correct collection

    try:
        query_embedding = embedding_model.get_embeddings([query_text])[0].values
    except Exception as e:
        logger.error(f"Failed to generate embedding for query: '{query_text}'. Error: {e}")
        mongo_client.close()
        return []

    if not query_embedding:
        logger.error(f"Failed to generate embedding for query: '{query_text}'")
        mongo_client.close()
        return []

    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",
                "path": "embedding",
                "queryVector": query_embedding,
                "numCandidates": 150,
                "limit": limit,
            }
        },
        {
            "$project": {
                "_id": 0,
                "game_id": 1,
                "play_id": 1,
                "official_date": 1,
                "home_team_name": 1,
                "away_team_name": 1,
                "venue_name": 1,
                "inning": 1,
                "half_inning": 1,
                "event": 1,
                "batter_name": 1,
                "pitcher_name": 1,
                "rich_text": 1,  # Return the full text
                "score": {"$meta": "vectorSearchScore"},
            }
        }
    ]
    try:
        results = list(collection.aggregate(pipeline))
        mongo_client.close()
        return results
    except Exception as e:
        logger.error(f"Error querying MongoDB: {e}")
        mongo_client.close()
        return []

# --- Main Execution and Update Logic ---

def update_mlb_data():
    """Fetches new game data, processes it, generates embeddings, and updates BigQuery and MongoDB."""
    logger.info("Starting MLB data update process...")
    create_game_events_table()  # Ensure BigQuery table exists

    for team_id in TEAMS.values():  # Iterate through all team IDs
        try:
            recent_games = get_team_games(team_id)
            for game_info in recent_games:
                game_pk = game_info['game_id']
                
                # Check if the game already exists in BigQuery
                query = f"SELECT game_id FROM `{PROJECT_ID}.mlb_data_2024.game_events` WHERE game_id = {game_pk} LIMIT 1"
                query_job = bq_client.query(query)
                results = list(query_job.result())
                
                if not results:  # Game doesn't exist, process it
                    logger.info(f"Processing new game: {game_pk}")
                    roster = get_roster_data(game_pk) #Get Roster
                    game_events = process_game_data(game_info, roster)
                    generate_embeddings_and_upload(game_events)
                else:
                    logger.info(f"Game {game_pk} already exists in BigQuery. Skipping.")

        except Exception as e:
            logger.error(f"Error updating data for team ID {team_id}: {e}")
            continue

    logger.info("MLB data update process completed.")


if __name__ == "__main__":
    # --- Initial setup (Uncomment if running for the first time)---
    update_mlb_data()  # Fetch initial data and create tables
    # --- Example Queries ---
    #After updating the database you should uncomment these queries to test functionality
    # query1 = "home runs by the Rangers against the Astros in the 9th inning"
    # results1 = query_mongodb(query1)
    # print(f"\nResults for query '{query1}':")
    # for result in results1:
    #     print(result)

    # query2 = "strikeouts at Yankee Stadium"
    # results2 = query_mongodb(query2)
    # print(f"\nResults for query '{query2}':")
    # for result in results2:
    #     print(result)

    # query3 = "games on 2024-05-15"  # Specific date query
    # results3 = query_mongodb(query3)
    # print(f"\nResults for query '{query3}':")
    # for result in results3:
    #     print(result)
    
    # # Example query 4: No Results
    query4 = "a goal scored by Messi"  # Irrelevant to MLB
    # results4 = query_mongodb(query4)
    # print(f"\nResults for query '{query4}':") #Should be empty.
    # for result in results4:
    # print(result)