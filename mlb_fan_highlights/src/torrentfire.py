import logging
import google.cloud.logging
from google.cloud.logging.handlers import CloudLoggingHandler
from google.cloud import secretmanager
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import pandas as pd
from google.cloud import bigquery
from typing import Dict, List, Optional
from vertexai.language_models import TextEmbeddingModel
import numpy as np


# --- Setup (Logging, Secret Manager, BigQuery Client) ---

def setup_logging():
    """Sets up Google Cloud Logging."""
    client = google.cloud.logging.Client()
    handler = CloudLoggingHandler(client)
    logger = logging.getLogger('mongodb_vector_search')
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    return logger

logger = setup_logging()

def get_secret(project_id, secret_id, version_id="latest", logger=None):
    """Retrieves a secret from Google Cloud Secret Manager."""
    try:
        client = secretmanager.SecretManagerServiceClient()
        name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"
        response = client.access_secret_version(request={"name": name})
        return response.payload.data.decode("UTF-8")
    except Exception as e:
        if logger:
            logger.error(f"Failed to retrieve secret {secret_id}: {str(e)}")
        raise

# Assuming you have your Google Cloud project ID set as an environment variable
PROJECT_ID = "gem-rush-007"  # Replace with your actual project ID
bq_client = bigquery.Client(project=PROJECT_ID)


# --- MongoDB Connection and Database Functions ---

def connect_to_mongodb(project_id: str = "gem-rush-007", secret_id: str = "mongodb-uri"):
    """Connects to MongoDB using URI from Secret Manager."""
    try:
        logger.info("Starting MongoDB connection process")
        uri = get_secret(project_id, secret_id, logger=logger)
        logger.info("Retrieved MongoDB URI from Secret Manager")
        client = MongoClient(uri, server_api=ServerApi('1'))
        client.admin.command('ping')  # Test connection
        logger.info("Successfully connected to MongoDB!")
        return client
    except Exception as e:
        logger.error(f"Error connecting to MongoDB: {str(e)}", exc_info=True)
        raise

def create_mongodb_database(client: MongoClient, db_name: str):
    """Creates a MongoDB database if it doesn't exist."""
    if db_name not in client.list_database_names():
        db = client[db_name]
        db.dummy_collection.insert_one({"dummy": "data"}) # Forces DB creation
        logger.info(f"MongoDB database '{db_name}' created.")
        db.dummy_collection.drop()
        logger.info("Dummy collection dropped.")
    else:
        logger.info(f"MongoDB database '{db_name}' already exists.")
    return client[db_name]


# --- Data Retrieval, Embedding Generation, and MongoDB Insertion ---

def get_bigquery_data_for_team(team_name: str) -> pd.DataFrame:
    """Retrieves combined data for a team from BigQuery."""
    dataset_id = f"{team_name}_mlb_data_2024"
    #  Improved query:  More robust handling of missing data, explicit joins.
    query = f"""
    SELECT
        g.official_date,
        g.game_id,
        g.home_team_name,
        g.away_team_name,
        g.home_score,
        g.away_score,
        p.inning,
        p.half_inning,
        p.event,
        p.event_type,
        COALESCE(p.description, 'No description available') AS description,  -- Handle null descriptions
        p.rbi,
        p.is_scoring_play,
        r_batter.full_name AS batter_name,
        r_pitcher.full_name AS pitcher_name,
        ps.at_bats,
        ps.hits,
        ps.home_runs,
        ps.walks,
        ps.strikeouts
    FROM
        `{PROJECT_ID}.{dataset_id}.games` AS g
    INNER JOIN
        `{PROJECT_ID}.{dataset_id}.plays` AS p
        ON g.game_id = p.game_id
    LEFT JOIN
        `{PROJECT_ID}.{dataset_id}.roster` AS r_batter
        ON p.batter_id = r_batter.player_id
    LEFT JOIN
        `{PROJECT_ID}.{dataset_id}.roster` AS r_pitcher
        ON p.pitcher_id = r_pitcher.player_id
    LEFT JOIN
        `{PROJECT_ID}.{dataset_id}.player_stats` AS ps
        ON p.game_id = ps.game_id AND p.batter_id = ps.player_id
    """

    try:
        df = bq_client.query(query).to_dataframe()
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = df[col].astype(str)  # Convert dates to strings
        return df
    except Exception as e:
        logger.error(f"Error retrieving data from BigQuery for {team_name}: {e}")
        return pd.DataFrame()


def generate_embeddings(df: pd.DataFrame, text_column: str = "description") -> pd.DataFrame:
    """Generates text embeddings using Vertex AI."""
    model = TextEmbeddingModel.from_pretrained("textembedding-gecko@003")

    def _embed_text(text: str) -> List[float]:
        try:
            if pd.isna(text) or text == 'No description available':  # Handle missing values
                return []
            embeddings = model.get_embeddings([text])
            return embeddings[0].values
        except Exception as e:
            logger.error(f"Embedding failed for text: '{text}'. Error: {e}")
            return []

    df['embedding'] = df[text_column].apply(_embed_text)
    return df

def insert_data_with_embeddings(db, collection_name: str, df: pd.DataFrame):
    """Inserts data with embeddings into MongoDB."""
    collection = db[collection_name]
    if collection.count_documents({}) > 0:
        logger.warning(f"Collection '{collection_name}' already exists. Appending data.")

    records = df.to_dict("records")
    for record in records:
        if not record['embedding']:
            logger.warning(f"Skipping record due to empty embedding: {record}")
            continue
        try:
            collection.insert_one(record)
        except Exception as e:
            logger.error(f"Error inserting record: {e}")
            # Consider handling specific errors (e.g., duplicate keys)

    logger.info(f"Data inserted/appended to MongoDB collection: {collection_name}")

def create_atlas_vector_search_index(client: MongoClient, db_name: str, collection_name: str):
    """Creates an Atlas Vector Search index if it doesn't exist."""
    index_name = "vector_index"
    db = client[db_name]
    collection = db[collection_name]

    existing_indexes = list(collection.list_indexes())
    if any(index['name'] == index_name for index in existing_indexes):
        logger.info(f"Index '{index_name}' already exists on {db_name}.{collection_name}.")
        return

    index_definition = {
        "name": index_name,
        "type": "vectorSearch",
        "fields": [
            {
                "type": "vector",
                "path": "embedding",
                "numDimensions": 768,
                "similarity": "dotProduct",
            }
        ]
    }
    try:
        result = collection.create_index([("embedding", "vector")], **index_definition)
        logger.info(f"Created Atlas Vector Search index: {result}")
    except Exception as e:
        logger.error(f"Error creating index: {e}")
        raise


# --- Querying with Embeddings ---

def query_mongodb_with_embeddings(db, collection_name: str, query_text: str, limit: int = 5) -> List[Dict]:
    """Queries MongoDB using vector similarity."""
    model = TextEmbeddingModel.from_pretrained("textembedding-gecko@003")
    try:
        query_embedding = model.get_embeddings([query_text])[0].values
    except Exception as e:
        logger.error(f"Failed to generate embedding for query: '{query_text}'. Error: {e}")
        return []

    if not query_embedding:
        logger.error(f"Failed to generate embedding for query: '{query_text}'")
        return []

    collection = db[collection_name]
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
                "official_date": 1,
                "game_id": 1,
                "home_team_name": 1,
                "away_team_name": 1,
                "description": 1,
                "event": 1,
                "batter_name": 1,
                "pitcher_name": 1,
                "score": {"$meta": "vectorSearchScore"},
            }
        }
    ]
    try:
        results = list(collection.aggregate(pipeline))
        return results
    except Exception as e:
        logger.error(f"Error querying MongoDB: {e}")
        return []



def local_scann_search(searcher, query_text: str, limit : int = 5):
        model = TextEmbeddingModel.from_pretrained("textembedding-gecko@003")

        try:
            query_embedding = model.get_embeddings([query_text])[0].values
        except Exception as e:
            logger.error(f"Failed to generate embedding for query: '{query_text}'. Error: {e}")
            return []
        if not query_embedding:
            logger.error(f"Failed to generate embedding for query: '{query_text}'")
            return []
        neighbors, distances = searcher.search(query_embedding, final_num_neighbors=limit)
        return neighbors, distances

# --- Main Execution (Example Usage) ---

if __name__ == "__main__":
    # 1. Connect to MongoDB and create the database (if needed)
    client = connect_to_mongodb()
    db_name = "mlb_data"  # Choose a database name
    db = create_mongodb_database(client, db_name)

    # 2. Process data for a specific team (e.g., the Rangers)
    team_name = "rangers"
    collection_name = f"{team_name}_plays"  # Collection for plays

    # 3. Get data from BigQuery
    df = get_bigquery_data_for_team(team_name)
    if df.empty:
        logger.error(f"No data retrieved from BigQuery for {team_name}.")
        exit()  # Exit if no data

    # 4. Generate embeddings
    df = generate_embeddings(df)

    # 5. Insert data into MongoDB
    insert_data_with_embeddings(db, collection_name, df)

    # 6. Create the Atlas Vector Search index
    create_atlas_vector_search_index(client, db_name, collection_name)

    # 7. Example query
    query = "a home run in the bottom of the ninth"
    results = query_mongodb_with_embeddings(db, collection_name, query)
    print(f"\nResults for query '{query}':")
    for result in results:
        print(result)

    # Example query 2
    query2 = "a strikeout with bases loaded"
    results2 = query_mongodb_with_embeddings(db, collection_name, query2)
    print(f"\nResults for query '{query2}':")
    for result in results2:
        print(result)

    # Example query 3: No Results
    query3 = "a goal scored by Messi"  # Irrelevant to MLB
    results3 = query_mongodb_with_embeddings(db, collection_name, query3)
    print(f"\nResults for query '{query3}':") #Should be empty.
    for result in results3:
        print(result)

    # Clean up (optional):  Close the MongoDB connection
    client.close()