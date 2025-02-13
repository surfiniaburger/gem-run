from datetime import datetime, date
import logging
import google.cloud.logging
from google.cloud.logging.handlers import CloudLoggingHandler
from google.cloud import secretmanager
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import pandas as pd
from google.cloud import bigquery
from typing import Dict, List, Optional
from vertexai.language_models import TextEmbeddingModel  # For embeddings
import numpy as np
from pymongo.operations import SearchIndexModel
import streamlit as st
import json


# --- LangChain Imports ---
from langchain_community.document_loaders import PyPDFLoader, BigQueryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_community.embeddings import VertexAIEmbeddings # Use VertexAI embeddings
from langchain_google_vertexai import ChatVertexAI
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain.chains import RetrievalQA
from langchain.schema import StrOutputParser, Document
from langchain.tools import StructuredTool
import json
import os

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
# Fetch secrets for consistent use
MONGO_URI = get_secret(PROJECT_ID, "mongodb-uri", logger=logger)

TEAMS = [
    'rangers', 'angels', 'astros', 'rays', 'blue_jays', 'yankees',
    'orioles', 'red_sox', 'twins', 'white_sox', 'guardians', 'tigers',
    'royals', 'padres', 'giants', 'diamondbacks', 'rockies', 'phillies',
    'braves', 'marlins', 'nationals', 'mets', 'pirates', 'cardinals',
    'brewers', 'cubs', 'reds', 'athletics', 'mariners', 'dodgers'
]

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
    """
    Helper function to construct the table name from a team's full name.

    Args:
        team_name (str): The full team name (e.g., "Minnesota Twins", "Arizona Diamondbacks")

    Returns:
        str: The formatted table name (e.g., "`gem-rush-007.twins_mlb_data_2024`")
    """
    # Convert to lowercase for consistent matching
    cleaned_name = team_name.lower().strip()

    # Try to find the team in the full names mapping
    if cleaned_name in FULL_TEAM_NAMES:
        team_key = FULL_TEAM_NAMES[cleaned_name]
        return f"`gem-rush-007.{team_key}_mlb_data_2024`"

    # If the exact full name isn't found, try to match with the team key directly
    for team_key in TEAMS:
        if team_key in cleaned_name:
            return f"`gem-rush-007.{team_key}_mlb_data_2024`"

    # If no match is found, return unknown table name
    return f"`gem-rush-007.unknown_team_mlb_data_2024`"

# --- MongoDB Connection and Database Functions ---

def connect_to_mongodb(uri: str = MONGO_URI):
    """Connects to MongoDB using URI from Secret Manager."""
    try:
        logger.info("Starting MongoDB connection process")
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


# --- BigQuery Helper Functions ---
#These now return Langchain Documents.

def fetch_team_games(team_name: str, limit: int = 2, specific_date: Optional[str] = None) -> List[Document]:
    """Fetches and returns game data as LangChain Documents."""
    team_id = TEAMS[get_team_key(team_name)]
    table_name = _get_table_name(team_name)

    query = f"""
        SELECT
            g.game_id,
            g.official_date,
            g.home_team_id,
            g.home_team_name,
            g.away_team_id,
            g.away_team_name,
            g.home_score,
            g.away_score,
            g.venue_name,
            g.status,
            {team_name}_win as team_win,
            {team_name}_margin as team_margin,
            subquery.max_end_time
        FROM
            {table_name}.games AS g
        INNER JOIN
            (SELECT
                game_id,
                MAX(end_time) AS max_end_time
            FROM
                {table_name}.plays
    """
    if specific_date:
        query += f" WHERE DATE(end_time) = @specific_date"
    query += f"""
            GROUP BY game_id
            ) AS subquery
            ON g.game_id = subquery.game_id
        WHERE
            (g.home_team_id = {team_id} OR g.away_team_id = {team_id})
    """
    if specific_date:
        query += f" AND g.official_date = @specific_date"
    query += " ORDER BY subquery.max_end_time DESC LIMIT @limit"

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("limit", "INT64", limit),
            bigquery.ScalarQueryParameter("specific_date", "DATE", specific_date),
        ]
    )
    try:
        query_job = bq_client.query(query, job_config=job_config)
        results = list(query_job.result())
        documents = []
        for row in results:
            row_dict = dict(row)
            content = f"Game ID: {row_dict.get('game_id', 'N/A')}, Date: {row_dict.get('official_date', 'N/A')}, " \
                      f"Home: {row_dict.get('home_team_name', 'N/A')} ({row_dict.get('home_score', 'N/A')}), " \
                      f"Away: {row_dict.get('away_team_name', 'N/A')} ({row_dict.get('away_score', 'N/A')}), " \
                      f"Status: {row_dict.get('status', 'N/A')}"
            metadata = {k: v for k, v in row_dict.items() if k != 'content'}
            metadata['source'] = 'bigquery'
            documents.append(Document(page_content=content, metadata=metadata))
        return documents

    except Exception as e:
        logger.error(f"Error in fetch_team_games for {team_name}: {e}")
        return []


def fetch_team_player_stats(team_name: str, limit: int = 100, specific_date: Optional[str] = None) -> List[Document]:
    """Fetches and returns player stats as LangChain Documents."""
    team_id = TEAMS[get_team_key(team_name)]
    table_name = _get_table_name(team_name)

    query = f"""
       SELECT
            ps.player_id,
            r.full_name,
            g.official_date as game_date,
            ps.at_bats,
            ps.hits,
            ps.home_runs,
            ps.rbi,
            ps.walks,
            ps.strikeouts,
            ps.batting_average,
            ps.on_base_percentage,
            ps.slugging_percentage
        FROM
            {table_name}.player_stats AS ps
        JOIN
            {table_name}.roster AS r
            ON ps.player_id = r.player_id
        INNER JOIN
            {table_name}.games AS g
            ON ps.game_id = g.game_id
        INNER JOIN (
            SELECT
                game_id,
                MAX(end_time) as max_end_time
            FROM
                {table_name}.plays
    """
    if specific_date:
        query += f" WHERE DATE(end_time) = @specific_date"
    query += f"""
            GROUP BY game_id
        ) AS subquery
        ON g.game_id = subquery.game_id
        WHERE
            (g.home_team_id = {team_id} OR g.away_team_id = {team_id})
    """
    if specific_date:
        query += f" AND g.official_date = @specific_date"
    query += f"""
       ORDER BY subquery.max_end_time DESC
       LIMIT @limit
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("limit", "INT64", limit),
            bigquery.ScalarQueryParameter("specific_date", "DATE", specific_date)
        ]
    )
    try:
        query_job = bq_client.query(query, job_config=job_config)
        results = list(query_job.result())
        documents = []
        for row in results:
            row_dict = dict(row)
            content = f"Player: {row_dict.get('full_name', 'N/A')}, Game Date: {row_dict.get('game_date', 'N/A')}, " \
                      f"AB: {row_dict.get('at_bats', 'N/A')}, H: {row_dict.get('hits', 'N/A')}, " \
                      f"HR: {row_dict.get('home_runs', 'N/A')}, RBI: {row_dict.get('rbi', 'N/A')}"
            metadata = {k: v for k, v in row_dict.items() if k != 'content'}
            metadata['source'] = 'bigquery'
            documents.append(Document(page_content=content, metadata=metadata))
        return documents
    except Exception as e:
        logger.error(f"Error in fetch_team_player_stats for {team_name}: {e}")
        return []


def fetch_team_plays(team_name: str, limit: int = 100, specific_date: Optional[str] = None) -> List[Document]:
    """Fetches and returns play data as LangChain Documents."""
    team_id = TEAMS[get_team_key(team_name)]
    table_name = _get_table_name(team_name)

    query = f"""
    SELECT
        p.play_id,
        p.inning,
        p.half_inning,
        p.event,
        p.event_type,
        p.description,
        p.rbi,
        p.is_scoring_play,
        r_batter.full_name as batter_name,
        r_pitcher.full_name as pitcher_name,
        p.start_time,
        p.end_time
    FROM
        {table_name}.plays AS p
    LEFT JOIN
        {table_name}.roster as r_batter
        ON p.batter_id = r_batter.player_id
    LEFT JOIN
        {table_name}.roster as r_pitcher
        ON p.pitcher_id = r_pitcher.player_id
    INNER JOIN
        {table_name}.games AS g
        ON p.game_id = g.game_id
    WHERE
        g.home_team_id = {team_id} OR g.away_team_id = {team_id}
    """
    if specific_date:
        query += f" AND g.official_date = @specific_date AND DATE(p.start_time) = @specific_date"
    query += f"""
    ORDER BY
        p.end_time DESC
    LIMIT @limit
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("limit", "INT64", limit),
            bigquery.ScalarQueryParameter("specific_date", "DATE", specific_date)
        ]
    )
    try:
        query_job = bq_client.query(query, job_config=job_config)
        results = list(query_job.result())
        documents = []
        for row in results:
            row_dict = dict(row)
            content = f"Play: {row_dict.get('description', 'N/A')}, Inning: {row_dict.get('inning', 'N/A')}, " \
                      f"Event: {row_dict.get('event', 'N/A')}, Batter: {row_dict.get('batter_name', 'N/A')}, " \
                      f"Pitcher: {row_dict.get('pitcher_name', 'N/A')}"
            metadata = {k: v for k, v in row_dict.items() if k != 'content'}
            metadata['source'] = 'bigquery'
            documents.append(Document(page_content=content, metadata=metadata))
        return documents

    except Exception as e:
        logger.error(f"Error in fetch_team_plays for {team_name}: {e}")
        return []


def fetch_player_plays(player_name: str, team_name: str, limit: int = 100, specific_date: Optional[str] = None) -> list:
    """
    Fetches play-by-play data for a specific player and returns as Langchain Documents

    Args:
        player_name (str): Full name of the player.
        team_name (str): Team name from TEAMS dictionary.
        limit (int, optional): Maximum number of plays to return. Defaults to 100.
        specific_date (str, optional): A specific date in 'YYYY-MM-DD' format to filter games.

    Returns:
        list: A list of dictionaries, each containing play details.
    """

    team_id = TEAMS[get_team_key(team_name)]
    table_name = _get_table_name(team_name)

    try:
        query = f"""
        SELECT
            p.play_id,
            p.inning,
            p.half_inning,
            p.event,
            p.event_type,
            p.description,
            p.start_time,
            g.official_date as game_date
        FROM
            {table_name}.plays AS p
        INNER JOIN 
            {table_name}.games AS g 
            ON p.game_id = g.game_id
        INNER JOIN
            {table_name}.roster AS r
            ON (p.batter_id = r.player_id OR p.pitcher_id = r.player_id)
        WHERE
            r.full_name = @player_name
            AND (g.home_team_id = {team_id} OR g.away_team_id = {team_id})
        """

        if specific_date:
            query += f" AND g.official_date = @specific_date AND DATE(p.start_time) = @specific_date"

        query += f"""
        ORDER BY 
            p.end_time DESC,
            p.start_time ASC
        LIMIT @limit
        """

        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("player_name", "STRING", player_name),
                bigquery.ScalarQueryParameter("limit", "INT64", limit),
                bigquery.ScalarQueryParameter("specific_date", "DATE", specific_date)
            ]
        )

        bq_client = bigquery.Client()
        query_job = bq_client.query(query, job_config=job_config)
        results = list(query_job.result())
        documents = []
        for row in results:
            row_dict = dict(row)
            content = f"Play: {row_dict.get('description', 'N/A')}, Inning: {row_dict.get('inning', 'N/A')}, " \
                      f"Event: {row_dict.get('event', 'N/A')}, Game Date: {row_dict.get('game_date', 'N/A')}"
            metadata = {k: v for k, v in row_dict.items() if k != 'content'}
            metadata['source'] = 'bigquery'
            documents.append(Document(page_content=content, metadata=metadata))

        return documents

    except Exception as e:
        logger.error(f"Error in fetch_player_plays for {player_name} on {team_name}: {e}")
        return []


# --- Data Retrieval, Embedding Generation, and MongoDB Insertion (BigQuery) ---
#These are now integrated within the main application flow.

def get_bigquery_data_for_team(team_name: str) -> pd.DataFrame:
    """Retrieves combined data for a team from BigQuery.  Returns a DataFrame."""
    dataset_id = f"{team_name}_mlb_data_2024"
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
        COALESCE(p.description, 'No description available') AS description,
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


def generate_embeddings_df(df: pd.DataFrame, text_column: str = "description") -> pd.DataFrame:
    """Generates text embeddings using Vertex AI *and returns a DataFrame*."""
    model = TextEmbeddingModel.from_pretrained("text-embedding-005")

    def _embed_text(text: str) -> List[float]:
        try:
            if pd.isna(text) or text == 'No description available':
                return []
            embeddings = model.get_embeddings([text])
            return embeddings[0].values
        except Exception as e:
            logger.error(f"Embedding failed for text: '{text}'. Error: {e}")
            return []

    df['embedding'] = df[text_column].apply(_embed_text)
    return df

def insert_data_with_embeddings(db, collection_name: str, df: pd.DataFrame):
    """Inserts data with embeddings into MongoDB, converting dates to strings."""
    collection = db[collection_name]
    if collection.count_documents({}) > 0:
        logger.warning(f"Collection '{collection_name}' already exists. Appending data.")

    for col in df.columns:
       if pd.api.types.is_datetime64_any_dtype(df[col]):
           df[col] = df[col].astype(str)
       elif isinstance(df[col].iloc[0], (pd.Timestamp, datetime, date)):
            df[col] = df[col].apply(lambda x: x.strftime('%Y-%m-%d') if isinstance(x, (date, datetime, pd.Timestamp)) else x)
    records = df.to_dict("records")
    for record in records:
        if not record['embedding']:
            logger.warning(f"Skipping record due to empty embedding: {record}")
            continue
        try:
            collection.insert_one(record)
        except Exception as e:
            logger.error(f"Error inserting record: {e}")


def create_atlas_vector_search_index(client: MongoClient, db_name: str, collection_name: str):
    """
    Creates an Atlas Search index (for vector search) if it doesn't already exist.
    """
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

    search_index_model = SearchIndexModel(
        definition=index_definition,
        name="vector_index"
    )

    try:
        result = collection.create_search_indexes(models=[search_index_model])
        logger.info(f"Created Atlas Search index: {result}")
    except Exception as e:
        logger.error(f"Error creating search index: {e}")
        raise


def process_all_teams(client, db_name):
    """Processes and inserts data for all MLB teams into MongoDB."""
    db = create_mongodb_database(client, db_name)
    for team_name in TEAMS:
        try:
            logger.info(f"Processing data for {team_name}")
            df = get_bigquery_data_for_team(team_name)
            if df.empty:
                logger.warning(f"No data retrieved from BigQuery for {team_name}. Skipping.")
                continue
            df = generate_embeddings_df(df)  # Use the DataFrame version
            collection_name = f"{team_name}_plays"
            insert_data_with_embeddings(db, collection_name, df)
            create_atlas_vector_search_index(client, db_name, collection_name)
            logger.info(f"Successfully processed {team_name}")
        except Exception as e:
            logger.error(f"Error processing {team_name}: {e}")
    logger.info("Completed processing all teams")


# --- PDF Processing (LangChain) ---

def process_pdf_data(pdf_files: list, client: MongoClient, db_name: str = "mlb_data", temp_collection_name: str = "temp_pdf_data"):
    """Processes uploaded PDFs, extracts text, generates embeddings (using VertexAI), and stores in MongoDB."""
    all_documents = []
    for pdf_file in pdf_files:
        try:
            loader = PyPDFLoader(pdf_file)
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=150,
                length_function=len  # Use for counting characters
            )
            split_documents = text_splitter.split_documents(documents)
            for doc in split_documents:
                doc.metadata['source'] = "pdf"  # Add source
            all_documents.extend(split_documents)

        except Exception as e:
            logging.error(f"Error processing PDF {pdf_file}: {e}")
            return None

    if not all_documents:
        logging.error("No documents extracted from PDFs.")
        return None

    try:
        db = client[db_name]
        collection = db[temp_collection_name]
        # Convert Langchain Documents to dictionaries with embeddings
        embeddings_model = VertexAIEmbeddings(model_name="text-embedding-005")

        texts = [doc.page_content for doc in all_documents]
        embeddings = embeddings_model.embed_documents(texts)

        documents_with_embeddings = []
        for i, doc in enumerate(all_documents):
          doc_dict = {
              "page_content": doc.page_content,
              "metadata": doc.metadata,
              "embedding": embeddings[i]  # Add the embedding
          }
          documents_with_embeddings.append(doc_dict)

        collection.insert_many(documents_with_embeddings)
        create_atlas_vector_search_index(client, db_name, temp_collection_name)

        vector_store = MongoDBAtlasVectorSearch(
                collection,
                embeddings_model,
                index_name = "vector_index"
                )

        return vector_store #return the vector store, to be used by retrievalQA

    except Exception as e:
        logging.error(f"Error creating vector store: {e}")
        return None

# --- Data Combination ---
def combine_data(bigquery_results: List[Document], pdf_vectorstore, query: str):
    """Combines BigQuery results with results from a PDF vectorstore query.

      Args:
        bigquery_results: List of Langchain Documents from Bigquery
        pdf_vectorstore: Vectorstore to get pdf results
        query: user query

      Returns:
        Combined list of documents.
    """
    combined_docs = [] + bigquery_results  # Start with BigQuery results

    if pdf_vectorstore:
      pdf_docs = pdf_vectorstore.similarity_search(
            query,
            k=5 #number of docs to retrieve
            )
      combined_docs.extend(pdf_docs)

    return combined_docs


# --- LangChain Setup ---
def setup_langchain(available_tools: List[StructuredTool]):
    """Sets up the LangChain LLM, prompt template, and RetrievalQA chain."""
    llm = ChatVertexAI(model_name="gemini-2.0-pro-exp-02-05", temperature=0.1, convert_system_message_to_human=True)

    system_template = """You are an expert sports podcast script generator.  Your task is to generate a JSON array of script entries, where each entry has a "speaker" key and a "text" key.

    **Example Output (IMPORTANT - Follow this EXACTLY):**

    ```json
    [
      {{
        "speaker": "Host",
        "text": "Welcome to the MLB podcast!"
      }},
      {{
        "speaker": "Analyst",
        "text": "Today we're discussing the recent Rangers game."
      }}
    ]
    ```

    **Rules:**

    *   Output MUST be a valid JSON array.
    *   Do NOT include any Markdown code blocks (```json ... ```).
    *   Do NOT include any introductory or explanatory text.  Output *only* the JSON array.
    *   Each object in the array MUST have a "speaker" key and a "text" key.
    """

    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template("Context:\n\n{context}"),
        HumanMessagePromptTemplate.from_template("Question: {question}"),
        HumanMessagePromptTemplate.from_template("Output the script as a single JSON array, with 'speaker' and 'text' keys for each object."),
    ])

    return llm, prompt


# --- Main Podcast Generation Function ---

def generate_mlb_podcasts(contents: str, data_source: int, client: MongoClient, pdf_files: list = None) -> dict:
    """
    Generates MLB podcast scripts.
    """
    db_name = "mlb_data"
    bigquery_documents = []  # This will now hold LangChain Documents
    pdf_vectorstore = None

    # --- Data Retrieval (Based on User Choice) ---
    if data_source == 1:
        # BigQuery Only (using BigQueryLoader)
        team_name = get_team_key(contents)
        if team_name:
            # Construct a combined query
            dataset_id = f"{team_name}_mlb_data_2024"
            base_query = f"""
                SELECT
                    g.official_date,
                    g.game_id,
                    g.home_team_name,
                    g.away_team_name,
                    CONCAT(
                        'Game on ', g.official_date, ' between ', g.home_team_name, ' and ', g.away_team_name, '. ',
                        'Inning: ', CAST(p.inning AS STRING), ', ', p.half_inning, '. ',
                        COALESCE(p.description, 'No description available'), ' ',
                        'Batter: ', COALESCE(r_batter.full_name, 'Unknown Batter'), ', ',
                        'Pitcher: ', COALESCE(r_pitcher.full_name, 'Unknown Pitcher'), '. ',
                        'Event: ', COALESCE(p.event, 'Unknown Event'), '. ',
                        'Home Score: ', CAST(g.home_score AS STRING), ', ',
                        'Away Score: ', CAST(g.away_score AS STRING), '.'
                    ) AS content,
                    g.official_date as game_date,  --Keep date for prompt.
                    g.home_team_name,
                    g.away_team_name,
                    p.inning,
                    p.half_inning,
                    p.description,
                    p.event,
                    r_batter.full_name as batter_name,
                    r_pitcher.full_name as pitcher_name

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
                WHERE (g.home_team_id = {TEAMS[team_name]} OR g.away_team_id = {TEAMS[team_name]})
            """
             # Add more conditions to the WHERE clause based on 'contents'.
            if "plays" in contents.lower():
                # Add play-specific filters
                pass # Add more specific filters.

            if "classic" in contents.lower():
                #Add filter to get important plays
                base_query += " AND p.is_scoring_play = true"

            base_query += " ORDER BY g.official_date DESC, p.inning DESC, p.half_inning DESC"
            base_query += " LIMIT 10" #limit it


            # Use BigQueryLoader
            loader = BigQueryLoader(base_query, project=PROJECT_ID)
            bigquery_documents = loader.load()
            #Add source
            for doc in bigquery_documents:
              doc.metadata['source'] = 'bigquery'


        # --- (Rest of data_source == 1 logic, if any) ---

    elif data_source == 2:
        # PDF Only
        if not pdf_files:
            return {"error": "No PDF files provided."}
        pdf_vectorstore = process_pdf_data(pdf_files, client, db_name)
        if not pdf_vectorstore:
            return {"error": "Failed to process PDF data."}

    elif data_source == 3:
        # Combined
        team_name = get_team_key(contents)
        if team_name:
            # --- BigQuery part (using BigQueryLoader) ---
            dataset_id = f"{team_name}_mlb_data_2024"
            base_query = f"""
                SELECT
                    g.official_date,
                    g.game_id,
                    g.home_team_name,
                    g.away_team_name,
                    CONCAT(
                        'Game on ', g.official_date, ' between ', g.home_team_name, ' and ', g.away_team_name, '. ',
                        'Inning: ', CAST(p.inning AS STRING), ', ', p.half_inning, '. ',
                        COALESCE(p.description, 'No description available'), ' ',
                        'Batter: ', COALESCE(r_batter.full_name, 'Unknown Batter'), ', ',
                        'Pitcher: ', COALESCE(r_pitcher.full_name, 'Unknown Pitcher'), '. ',
                        'Event: ', COALESCE(p.event, 'Unknown Event'), '. ',
                        'Home Score: ', CAST(g.home_score AS STRING), ', ',
                        'Away Score: ', CAST(g.away_score AS STRING), '.'
                    ) AS content,
                    g.official_date as game_date,
                    g.home_team_name,
                    g.away_team_name,
                    p.inning,
                    p.half_inning,
                    p.description,
                    p.event,
                    r_batter.full_name as batter_name,
                    r_pitcher.full_name as pitcher_name
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
                WHERE (g.home_team_id = {TEAMS[team_name]} OR g.away_team_id = {TEAMS[team_name]})
            """
            if "plays" in contents.lower():
                # Add play-specific filters
                pass  # Add more specific conditions here.

            if "classic" in contents.lower():
                #Add filter to get important plays
                base_query += " AND p.is_scoring_play = true"

            base_query += " ORDER BY g.official_date DESC, p.inning DESC, p.half_inning DESC"
            base_query += " LIMIT 10"

            loader = BigQueryLoader(base_query, project=PROJECT_ID)
            bigquery_documents = loader.load()
            #Add source
            for doc in bigquery_documents:
              doc.metadata['source'] = 'bigquery'

        if pdf_files:
            pdf_vectorstore = process_pdf_data(pdf_files, client, db_name)
            if not pdf_vectorstore:
                logging.warning("Failed to process PDF data, using BigQuery only.")

    # --- Combine Data (if applicable) ---

    if data_source == 3:
        combined_documents = combine_data(bigquery_documents, pdf_vectorstore, contents)
    elif data_source == 2:
        combined_documents = pdf_vectorstore.similarity_search(
            contents,
            k=5  # Number of documents to retrieve
        )
    else:  # data_source == 1
        combined_documents = bigquery_documents

    # --- Tool Definitions (for LangChain) ---

    def fetch_team_games_tool(team_name: str, limit: int = 5, specific_date: Optional[str] = None) -> str:
        """Fetches recent games for a team and returns as a JSON string."""
        try:
            games = fetch_team_games(team_name, limit, specific_date)
            return json.dumps([doc.dict() for doc in games]) # Convert Documents to dicts
        except Exception as e:
            logger.error(f"Error in fetch_team_games_tool: {e}")
            return f"Error: {e}"

    def fetch_player_stats_tool(team_name: str, limit: int = 5, specific_date: Optional[str] = None) -> str:
      """Fetches player stats and returns as a JSON string."""
      try:
        player_stats = fetch_team_player_stats(team_name, limit, specific_date)
        return json.dumps([doc.dict() for doc in player_stats])
      except Exception as e:
            logger.error(f"Error in fetch_player_stats_tool: {e}")
            return f"Error: {e}"


    def fetch_team_plays_tool(team_name: str, limit: int = 5, specific_date: Optional[str] = None) -> str:
      """Fetches team plays and returns as a JSON string."""
      try:
        plays = fetch_team_plays(team_name, limit, specific_date)
        return json.dumps([doc.dict() for doc in plays])  # Convert Documents to dicts

      except Exception as e:
            logger.error(f"Error in fetch_team_plays_tool: {e}")
            return f"Error: {e}"

    def fetch_player_plays_tool(player_name: str, team_name: str, limit: int = 100, specific_date: Optional[str] = None) -> str:
      """Fetches player specific plays data, and returns as JSON string"""
      try:
        plays = fetch_player_plays(player_name, team_name, limit, specific_date)
        return json.dumps([doc.dict() for doc in plays])
      except Exception as e:
            logger.error(f"Error in fetch_player_plays_tool: {e}")
            return f"Error: {e}"


    # Create StructuredTool objects
    team_games_tool = StructuredTool.from_function(fetch_team_games_tool)
    player_stats_tool = StructuredTool.from_function(fetch_player_stats_tool)
    team_plays_tool = StructuredTool.from_function(fetch_team_plays_tool)
    player_plays_tool = StructuredTool.from_function(fetch_player_plays_tool)


    available_tools = [team_games_tool, player_stats_tool, team_plays_tool, player_plays_tool]

    # --- LangChain Setup and Execution ---
    llm, prompt = setup_langchain(available_tools)

     # Build the RetrievalQA chain or LLMChain based on data_source
    if data_source == 2 or data_source == 3:
        retriever = pdf_vectorstore.as_retriever(search_kwargs={'k': 5})
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True,
        )
    elif data_source == 1:
        # Use LCEL for a simple prompt | llm chain
        chain = (
            {"context": lambda x: "\n\n".join(doc.page_content for doc in x["input_documents"]),
             "question": lambda x: x["query"]}
            | prompt
            | llm
            | StrOutputParser()  # Get a string output
        )

    else:
        raise ValueError("Invalid data source selected")


    try:
        if data_source == 1:
          #invoke the chain
          result = chain.invoke({"input_documents": combined_documents, "query": contents})

        else:
          result = qa_chain({"query": contents, "tools": available_tools})

        # --- JSON Validation and Processing ---
        try:
            #if it is not data source 1, it will be a dict, so get result
            if data_source != 1:
              result = result['result']

            podcast_script = json.loads(result)
            if not isinstance(podcast_script, list):
                raise ValueError("Output is not a JSON array")
            for item in podcast_script:
                if not ("speaker" in item and "text" in item):
                    raise ValueError("Invalid JSON object format")
            return podcast_script  # Return the validated JSON

        except (json.JSONDecodeError, ValueError) as e:
            logging.error(f"JSON validation error: {e}")
            return {"error": f"Invalid JSON output: {e}, Output: {result}"}

    except Exception as e:
        logging.error(f"Error generating podcast script: {e}")
        return {"error": str(e)}


# --- Main Execution (Example) ---
def get_user_data_source():
    print("Choose a data source:")
    print("1. Use Existing Structured Data (BigQuery)")
    print("2. Upload PDF Data")
    print("3. Combine Existing and Uploaded Data")

    while True:
        choice = input("Enter your choice (1, 2, or 3): ")
        if choice in ("1", "2", "3"):
            return int(choice)
        print("Invalid choice. Please enter 1, 2, or 3.")


if __name__ == "__main__":
    st.set_page_config(page_title="MLB Podcast Generator", page_icon=":baseball:")
    st.title("MLB Podcast Generator")

    client = connect_to_mongodb()  # Connect to MongoDB *once* at the beginning

    with st.sidebar:
        st.header("Data Source")
        data_source = st.radio(
            "Choose a data source:",
            options=[
                (1, "Use Existing Structured Data (BigQuery)"),
                (2, "Upload PDF Data"),
                (3, "Combine Existing and Uploaded Data"),
            ],
            format_func=lambda x: x[1],  # Display the label, not the number
        )
        data_source = data_source[0]  # Get the number (1, 2, or 3)

        pdf_files = []
        if data_source in (2, 3):
            uploaded_files = st.file_uploader("Upload PDF Files", type=["pdf"], accept_multiple_files=True)
            # Convert UploadedFile objects to file paths (temporary files)
            for uploaded_file in uploaded_files:
                with open(uploaded_file.name, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                pdf_files.append(uploaded_file.name)


    st.header("Podcast Request")
    user_query = st.text_area("Enter your MLB podcast request:", height=150)

    if st.button("Generate Podcast Script"):
        if not user_query:
            st.warning("Please enter a podcast request.")
        else:
            with st.spinner("Generating podcast script..."):
                try:
                    podcast_script = generate_mlb_podcasts(user_query, data_source, client, pdf_files)

                    if "error" in podcast_script:
                        st.error(f"Error: {podcast_script['error']}")
                    else:
                        st.subheader("Generated Podcast Script:")
                        # Display as formatted JSON (for now)
                        # st.json(podcast_script)

                        # Display in a more user-friendly way:
                        for entry in podcast_script:
                            speaker = entry['speaker']
                            text = entry['text']
                            st.markdown(f"**{speaker}:** {text}")


                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")
                finally:
                    # Clean up temporary files
                    for file_path in pdf_files:
                        try:
                            os.remove(file_path)
                        except Exception as e:
                            logging.error(f"Error deleting temp file {file_path}: {e}")


    client.close()  # Close MongoDB connection when Streamlit app closes


