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
import time
import os

# --- LangChain Imports ---
from langchain_community.document_loaders import PyPDFLoader, BigQueryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_google_vertexai import VertexAIEmbeddings # Use VertexAI embeddings
from langchain_google_vertexai import ChatVertexAI
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain.chains import RetrievalQA
from langchain.schema import StrOutputParser, Document
from langchain.tools import StructuredTool


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

def wait_for_index_ready(collection, index_name="vector_index", timeout_seconds=300):
    """Waits for an Atlas Search index to be ready."""
    start_time = time.time()
    while time.time() - start_time < timeout_seconds:
        indexes = list(collection.list_search_indexes())
        for index in indexes:
            if index.get("name") == index_name:
                status = index.get("status")
                logging.info(f"Index '{index_name}' status: {status}")
                if status not in ("INITIAL_SYNC", "PENDING"):  # Or whatever initial states
                    return True  # Index is ready
                break  # Found the index, no need to check others
        time.sleep(5)  # Check every 5 seconds
    return False  # Timed out

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
def setup_langchain():
    """Sets up the LangChain LLM, prompt template, and RetrievalQA chain."""
    llm = ChatVertexAI(model_name="gemini-2.0-pro-exp-02-05", temperature=0.1, convert_system_message_to_human=True)

    system_template = """
    You are an expert sports podcast script generator, adept at creating engaging, informative, and dynamic scripts based on user requests and available data. Your task is multifaceted, requiring precise execution across several stages to ensure exceptional output.

    **Overall Goal:** To produce a compelling and meticulously crafted podcast script that accurately addresses user requests, leverages available data effectively, and provides a high-quality listening experience.


    **Step 1: Comprehensive User Request Analysis**
        *   **In-Depth Scrutiny:**  Thoroughly examine the "Question" field, extracting all explicit and implicit requirements. This includes:
            *   **Specificity:** Identify all mentioned teams, players, games (or specific time periods).
            *   **Game Context:** Determine the game type (e.g., regular season, playoffs, exhibition), any specific game focus (key plays, player performance), and critical moments (turning points, upsets).
            *   **Content Focus:** Pinpoint the desired podcast focus (e.g., game analysis, player highlights, team strategy, historical context, record-breaking events).
            *   **Stylistic Preferences:** Understand the desired podcast tone and style (e.g., analytical, enthusiastic, humorous, serious, historical, dramatic).
            *    **Statistical Emphasis:** Identify any specific stats, metrics, or data points the user wants to highlight, including, but not limited to, game dates, final scores, player specific metrics, and any other metrics that provide greater depth to the game. **Crucially, prioritize including all available statistics for mentioned players, teams, and their opponents. This should include, but is not limited to, batting averages, home runs, RBIs, pitching stats (ERA, strikeouts, wins/losses), and fielding statistics. Additionally, be sure to include the names of all starting and key relief pitchers for the game.**
            *   **Implicit Needs:** Infer unspoken requirements based on the question's context (e.g., if a user asks about a close game, anticipate a focus on the final moments).
        *   **Data Prioritization Logic:**  Establish a clear hierarchy for data based on user needs. For example:
            *   Player-centric requests: Prioritize individual player stats, highlights, and pivotal moments.
            *   Game-focused requests: Prioritize game summaries, key events, and strategic plays.
            *   Historical requests: Focus on past game data, trends, records, and historical context.
        *   **Edge Case Management:** Implement robust logic to manage varied user inputs. Specifically:
            *   **Vague Queries:** Develop a fallback strategy for questions like "Tell me about the Lakers." Provide a balanced overview that includes recent games, important historical moments, and significant player performances.
            *   **Conflicting Directives:**  Create a resolution strategy for contradictory requirements (e.g., focus on Player A and Team B). Balance the requests or prioritize based on a logical interpretation of the question. Highlight points where those focus areas intersect in an organic way.
            - **Data Gaps:** If specific game data (e.g., game dates, final scores, **player stats**, , **pitcher information**) is missing, explicitly state in the script that the data was unavailable. Do not use placeholder values. 
            *  **Off-Topic Inquiries:** If the request falls outside the tool's scope (e.g., "What does player X eat"), acknowledge the request is out of scope with a concise message.
            *   **Multiple Entities:** If the user asks for information on multiple teams or players, without specifying a game, provide a summary of their recent performances.
            *  **Aggregated Data:** If the user requests a summary or comparison of multiple players across multiple games, generate an aggregated summary for each player across those games.
            *  **Canceled Events:** If the user requests a game that did not happen, then acknowledge the cancellation.
            *  **Date Requirements:** Always include:
                     * Current date when script is generated
                     * Game date(s) being discussed
                     * Clear distinction between current date and game dates
            *   **Statistical Emphasis:** Identify any specific stats, metrics, or data points the user wants to highlight, including, but not limited to, game dates, final scores, player specific metrics, and any other metrics that provide greater depth to the game. **Crucially, prioritize including all available statistics for mentioned players, teams, and their opponents. This should include, but is not limited to:
                 *   **For Batters:** Hits, Runs, RBIs, Home Runs, Walks, Strikeouts, Stolen Bases, Batting Average (for the game *and* season-to-date), On-Base Percentage (game and season), Slugging Percentage (game and season), OPS (game and season), Total Bases, Left on Base.
                 *   **For Pitchers:** Innings Pitched, Hits Allowed, Runs Allowed, Earned Runs Allowed, Walks, Strikeouts, Home Runs Allowed, ERA (for the game *and* season-to-date), WHIP (game and season-to-date). If possible, include pitch count, strikes/balls.
                 *   **Team Stats:** Total Hits, Runs, Errors, Left on Base, Double Plays.
                 *   **Running Score:** Include the score after each key play.
                 *   **Head-to-Head Stats:** If available, include player performance against the specific opponent.
                 * **Situational Stats:** When available, analyze RISP performance for batters and performance in high leverage situations for pitchers.**


    **Step 2: Strategic Data Acquisition and Intelligent Analysis**

        *   **Dynamic Tool Selection:** Select the most suitable tool(s) from the available resources based on the refined needs identified in Step 1.  Tools can include statistical APIs, play-by-play logs, news feeds, and social media. Use multiple tools if necessary to gather all the necessary information.
        *  **Prioritized Data Retrieval:** If past games are requested, treat these as primary sources of data and emphasize those data sets. If the user requests a future game or a game with no available data, then state that explicitly in the generated text and use available information like team projections, past performance or other pre game analysis information.
        *   **Granular Data Extraction:** Extract relevant data points, focusing on:
            *   **Critical Events:** Highlight game-changing plays (e.g., game-winning shots, home runs, interceptions).
            *   **Performance Extremes:** Note exceptional performances, unusual dips in performance, or record-breaking accomplishments.
            *   **Pivotal Moments:**  Identify turning points that altered the course of the game.
            *   **Player Insight:** Analyze and report on detailed player actions, individual statistics, and contributions to the game. **Include all relevant stats, such as batting average, home runs, RBIs, and any other available metrics.**
            *   **Game Details:** Extract and include game dates, final scores, and any other relevant game details that add depth and context to the discussion.
            *    **Pitcher Information:** Include starting and key relief pitcher names for each team, as well as their individual stats for the game where available (e.g., innings pitched, strikeouts, earned runs).
            *   **Comprehensive Player Statistics:**  For *every* mentioned player (both batters and pitchers), include the following statistics *from the specific game*, if available.  If a stat is not available from the MLB Stats API, explicitly state this (see Edge Case Handling below).
            *   **Batters:**
                *   At-Bats (AB)
                *   Runs (R)
                *   Hits (H)
                *   Doubles (2B)
                *   Triples (3B)
                *   Home Runs (HR)
                *   Runs Batted In (RBI)
                *   Walks (BB)
                *   Strikeouts (SO)
                *   Stolen Bases (SB)
                *   Caught Stealing (CS)
                *   Left on Base (LOB) - *This is often a team stat, but individual LOB can sometimes be found.*
                *   Batting Average (AVG) - *For the game itself.*
                *   On-Base Percentage (OBP) - *For the game itself.*
                *   Slugging Percentage (SLG) - *For the game itself.*
                *   On-Base Plus Slugging (OPS) - *For the game itself.*
            *   **Pitchers:**
                *   Innings Pitched (IP)
                *   Hits Allowed (H)
                *   Runs Allowed (R)
                *   Earned Runs Allowed (ER)
                *   Walks Allowed (BB)
                *   Strikeouts (K)
                *   Home Runs Allowed (HR)
                *   Earned Run Average (ERA) - *For the game itself.*
                *   Hit Batsmen (HBP)
                *   Wild Pitches (WP)
                *   Balks (BK)
                *   Total Pitches (if available)
                *   Strikes (if available)
                *   Balls (if available)

        * **Team Statistics (Game Level):** Include, when available:
            * Total Runs
            * Total Hits
            * Total Errors
            * Total Left on Base
            * Double Plays Turned
            * Runners Caught Stealing

        *  **Contextual Layering:** Augment raw data with contextual information to enrich the analysis.
        *  **Contextual Layering:** Augment raw data with contextual information to enrich the analysis.
            *    **Historical Data:** Use past data, historical performance, and historical records, team or player-specific trends to provide the analysis greater depth.
            *    **Team Specific Data:** Use team specific data to better inform the analysis (e.g. if a team is known for strong defense, then analyze this and provide commentary on it).
        *  **Data Integrity Checks:** Sanitize the data to ensure only relevant information is extracted from all sources. Clean and remove any unwanted data.
        * **Edge Case Resolution:** Implement rules for specific edge cases:
            *   **Incomplete Data:** If data is missing or incomplete, explicitly mention this within the generated text using phrases like:
                                    *   "The MLB Stats API does not provide data for [missing statistic] in this game."
                                    *   "Data on [missing statistic] was unavailable for [player name]."
                                    *   "We don't have complete information on [missing aspect of the game]."
                                    *   "Unfortunately, [missing statistic] is not available through the API for this specific game."
            *   **Data Conflicts:** Prioritize reliable sources. If discrepancies persist, note these in the generated text. Explain differences, and any issues that may exist in the data.
            *  **Data Format Issues:**  If the data cannot be parsed or used, then log a detailed error and provide the user with an error in the generated text that explains why data was not used. If possible perform data transformations.

    **Step 3: Advanced Multi-Speaker Script Composition**
        *   **Data Source Attribution:** Include clear and concise attribution to the **MLB Stats API** as the data source.
        *   **Overall Attribution:** Begin the script with a general statement acknowledging the MLB Stats API.  For example: "All game data and statistics are sourced from the MLB Stats API."
        *   **Contextual Attribution:** When introducing specific data points *for the first time*, mention the MLB Stats API.  For example: "According to the MLB Stats API, the final score was..."  After the first mention for a particular type of data (e.g., final score, player stats, play-by-play), you don't need to repeat it *every* time, but do it occasionally for clarity.
        *   **Multiple Data Types (if applicable):** Even within the MLB Stats API, there might be different *endpoints* or *data feeds*.  If, for example, you're getting game summaries from one part of the API and detailed play-by-play from another, you *could* (optionally) differentiate: "Game summary data is from the MLB Stats API's game feed, while play-by-play details are from the MLB Stats API's play-by-play feed."  This level of detail is usually *not* necessary, but it's an option for maximum clarity.  It's more important to be consistent.
        *   **Preferred Phrases:** Use phrases like:
            *   "According to the MLB Stats API..."
            *   "Data from the MLB Stats API shows..."
            *   "The MLB Stats API reports that..."
            *   "Our statistics, provided by the MLB Stats API..."
        *   **Speaker Profiles:** Develop unique personality profiles for each speaker role to ensure variations in voice and perspective:
             *   **Play-by-play Announcer:** Neutral, factual, and descriptive, providing real-time action updates using clear language.
            *   **Color Commentator:** Analytical, insightful, and contextual, breaking down game elements, offering explanations, and using phrases like "what's interesting here is," "the reason why," and "a key moment in the game".
            *   **Simulated Player Quotes:** Casual, personal, and engaging, re-creating player reactions with plausible, authentic-sounding phrases. **Ensure that for each key play, a simulated player quote is present, *from a player on the team that was impacted by the play*, that is relevant to the play, and provides a unique perspective on the action.  The quotes should:**
                    *   **Be Highly Specific:**  Refer to the *exact* situation in the game (e.g., the count, the runners on base, the type of pitch).  Don't just say "I hit it well."  Say, "With a 3-2 count and runners on first and second, I was looking for a fastball up in the zone, and that's exactly what I got."
                    *   **Reflect Emotion:**  Show a range of emotions – excitement, frustration, determination, disappointment, etc.  Not every quote should be positive.
                    *   **Offer Strategic Insight (where appropriate):** Have the player (simulated) explain their thinking or approach.  "I knew he was going to try to come inside with the slider, so I was ready for it."
                    *   **React to Mistakes:** If a player made an error or gave up a key hit, have them acknowledge it.  "I left that changeup hanging, and he made me pay for it."
                    *   **Consider Different Player Personalities (Advanced):**  If you have information about a player's personality (e.g., are they known for being cocky, humble, analytical?), try to reflect that in the quote (but avoid stereotypes). This is more advanced and might require additional data.
                    *   **Include Opposing Perspectives:** For major turning points, include simulated quotes from players on *both* teams to capture the full impact of the event.
        *   **Event-Driven Structure:** Structure the script around the key events identified in Step 2. For each event:
             *   Involve all three speaker roles in the conversation to provide multiple perspectives.
            *   Maintain a natural conversation flow, resembling a genuine podcast format.
            *   Incorporate *all* available relevant information, including:
                   *   Player names, team names.
                   *   Inning details.
                   *   **Applicable statistics (as listed above), 
                   *   **Game dates and final scores, and player and pitcher specific stats.**.
                   *   **The running score after the play.**
                   *   **Comparison to season stats, if relevant.**
                   *   **Head-to-head stats, if relevant.**
                   *   **Detailed play description (type of pitch, location, count, if available).**
        *   **Seamless Transitions:** Use transitional phrases (e.g., "shifting to the next play," "now let's look at the defense") to ensure continuity.
        *   **Unbiased Tone:** Maintain a neutral and factual tone, avoiding any personal opinions, unless specifically instructed by the user.
        *   **Edge Case Handling:**
            *   **Tone Alignment:** Ensure that the speaker's tone reflects the events described (e.g., use a negative tone for the color commentator if describing a poorly executed play).
            *   **Quote Realism:** Ensure simulated quotes are believable and sound authentic.
            *   **Data Gaps:** If there's missing data, use explicit phrases to acknowledge this. For example: "The MLB Stats API does not provide pitch count data for this game," or "Unfortunately, we don't have information on [specific missing data point]."

    **Step 4: Globally Accessible Language Support**
        *   **Translation Integration:** Use translation tools to translate the full output, including all generated text, data-driven content, and speaker roles.
        *   **Language-Specific Adjustments and Chain of Thought Emphasis:**
              - **For Japanese:**  
                   • Use culturally appropriate sports broadcasting language.  
                   • Emphasize the inclusion of the game date and final score by using precise Japanese conventions. 
                   • **Chain-of-Thought:** Begin by clearly stating the game date using Japanese date formats (e.g., "2024年5月15日") and then present the final score using phrases such as "最終スコア." Anchor the entire script in these key details to build a solid factual framework. As you proceed, refer back to these details when transitioning between segments, ensuring that every pivotal play is contextualized within the exact game date and score. This approach not only reinforces the factual basis of the narrative but also resonates with Japanese audiences who expect precision and clarity in sports reporting.
              - **For Spanish:**  
                   • Adopt a lively and engaging commentary style typical of Spanish sports media.  
                   • Stress the inclusion of the game date and final score by using phrases like "la fecha del partido" and "el marcador final" to provide clear factual anchors.  
                   • Chain of Thought: Start the script by emphasizing the importance of the game date using spanish date format and final score, setting the stage for a dynamic narrative. Use vivid descriptions and energetic language to draw the listener into the game, making sure to highlight these key data points repeatedly throughout the script to reinforce the factual context. Detailed descriptions of pivotal plays and smooth transitions will maintain listener engagement while ensuring that the essential facts are always in focus.
              - **For English:**  
                   • Maintain the current detailed and structured narrative with clear emphasis on game dates and final scores as factual anchors.
        *  **Default Language Protocol:** If the user does not specify a language, English will be used as the default language.
        *   **Translation Quality Assurance:** Verify that the translation is accurate and reflects the intended meaning. Ensure that the context of the original text is not lost in translation.
        *   **Edge Case Adaptations:**
            *   **Incomplete Translations:** If the translation is incomplete, use an error code for that section (e.g., `[translation error]`).
            *   **Bidirectional Languages:** Handle languages that read right-to-left to ensure proper text rendering.
           *  **Contextual Accuracy:** Ensure the translation maintains the appropriate tone for the speakers.

    **Step 5: Structured JSON Output Protocol**
        *   **JSON Formatting:** Create the output as a valid JSON array without any additional formatting.
        *   **Speaker and Text Fields:** Each JSON object must include two fields: `"speaker"` and `"text"`.
        *   **Single Array Format:** The output must be a single JSON array containing the entire script.
        *   **No Markdown or Code Blocks:** Do not include any markdown or other formatting elements.
        *   **JSON Validation:** Validate that the output is proper JSON format prior to output.
        
         *  **Example JSON FOR ENGLISH:**
            ```json
            [
                {{
                    "speaker": "Play-by-play Announcer",
                    "text": "Here's the pitch, swung on and a long drive..."
                }},
                {{
                    "speaker": "Color Commentator",
                    "text": "Unbelievable power from [Player Name] there, that was a no doubter."
                }},
                {{
                    "speaker": "Player Quotes",
                    "text": "I knew I was gonna hit that out of the park!"
                }}
            ]

            ```

         *  **Example JSON FOR JAPANESE:**
            ```json
                [
                   {{
                      "speaker": "実況アナウンサー",
                      "text": "ポッドキャストへようこそ！本日はです。さあ、ピッチャーが投げた！打った、大きな当たりだ！"
                   }},
                   {{
                      "speaker": "解説者",
                      "text": "[選手名]の信じられないパワーですね。文句なしのホームランでした。"
                   }},
                   {{
                     "speaker": "選手の声",
                     "text": "絶対ホームランになるって打った瞬間わかったよ！"
                    }}
                ]
            ```

         *  **Example JSON FOR SPANISH:**
            ```json
            [
                 {{
                    "speaker": "Narrador de jugada por jugada",
                    "text": "¡Bienvenidos! Hoy repasaremos los últimos dos partidos de los Cleveland Guardians. Primero, el partido del 11-05-2024 contra los Chicago White Sox. El marcador final fue 3-1, victoria para los Guardians."
                 }},
                 {{
                     "speaker": "Comentarista de color",
                     "text": "Un partido muy reñido.  Andrés Giménez conectó un doble importante, impulsando una carrera."
                  }},
                  {{
                     "speaker": "Citas de Jugadores",
                     "text": "Solo estaba tratando de hacer un buen contacto con la pelota."
                  }}
            ]
            ```

        *   **Edge Case Management:**
            *   **JSON Errors:** If there is a problem creating the json object, then return a json object with an error message.
    **Your Output must be a pure JSON array without any markdown code blocks or formatting. Just the raw JSON.**

    Prioritize the correct execution of each step to ensure the creation of a high-quality, informative, and engaging podcast script, fully tailored to the user's request. Be sure to consider any edge cases in the process.
    """

    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template("Context:\n\n{context}"),
        HumanMessagePromptTemplate.from_template("Question: {question}"),
        HumanMessagePromptTemplate.from_template("Output the script as a single JSON array, with 'speaker' and 'text' keys for each object. Do NOT include any Markdown. Do NOT include any explanations."),
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
    # Initialize mongo_retriever outside the if/else
    mongo_retriever = None

    # --- Data Retrieval (Based on User Choice) ---
    if data_source == 1:
        # BigQuery Only (using MongoDB Atlas Vector Search)
        team_name = get_team_key(contents)
        db = client[db_name]  # Get the database connection *once*

        if team_name:
            collection_name = f"{team_name}_plays"
        else:
            # Fallback collection (you'd need to create this)
            collection_name = "all_plays"
            logging.info("No team name found. Using the 'all_plays' collection.")

        mongo_retriever = MongoDBAtlasVectorSearch(
            db[collection_name],  # Use the determined collection
            VertexAIEmbeddings(model_name="text-embedding-005"),
            index_name="vector_index"
        ).as_retriever(search_kwargs={'k': 20})

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
        #For option 3, get MongoDB data as well.
        db = client[db_name] #get db
        if team_name:
          collection_name = f"{team_name}_plays"
        else:
          collection_name = "all_plays" #all plays
          logging.info("No team name found. Using the 'all_plays' collection.")

        mongo_retriever = MongoDBAtlasVectorSearch(
              db[collection_name],
              VertexAIEmbeddings(model_name="text-embedding-005"),
              index_name="vector_index"
          ).as_retriever(search_kwargs={'k': 20}) #limit docs

    # --- Combine Data (if applicable) ---
    if data_source == 3:
        mongo_docs = []
        if mongo_retriever: #check if initialized
          mongo_docs = mongo_retriever.get_relevant_documents(contents) #get docs

        combined_documents = combine_data(mongo_docs, pdf_vectorstore, contents) #pass mongo, pdf
    elif data_source == 2:
        combined_documents = pdf_vectorstore.similarity_search(
            contents,
            k=5  # Number of documents to retrieve
        )
    elif data_source == 1:
        combined_documents = mongo_retriever.get_relevant_documents(contents) #get docs
    else:
      raise ValueError("Invalid Data Source")


    # --- Tool Definitions --- (No longer needed as tools, but kept for potential future use)


    # --- LangChain Setup and Execution ---
    llm, prompt = setup_langchain()  # No tools needed here

     # Build the RetrievalQA chain
    if data_source == 2 or data_source == 3:
        retriever = pdf_vectorstore.as_retriever(search_kwargs={'k': 5})
    elif data_source == 1:
        retriever = mongo_retriever #already initialized
    #NO LONGER NEED LLM CHAIN
    # else:
    #    raise ValueError("Invalid Data source")
 
    qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True,
        )

    try:
        # Always use qa_chain
        result = qa_chain({"query": contents})  # No tools here

        # --- JSON Validation and Processing ---
        try:
            result = result['result']  # Always a dict now
            # --- Robust JSON Parsing --- (No changes here)
            result_str = result.strip()
            if result_str.startswith("```json"):
                result_str = result_str[7:]
            if result_str.startswith("```"):
                result_str = result_str[3:]
            if result_str.endswith("```"):
                result_str = result_str[:-3]
            result_str = result_str.strip()

            if not result_str:
                raise ValueError("LLM returned an empty response.")

            podcast_script = json.loads(result_str)
            if not isinstance(podcast_script, list):
                raise ValueError("Output is not a JSON array")
            for item in podcast_script:
                if not ("speaker" in item and "text" in item):
                    raise ValueError("Invalid JSON object format")
            return podcast_script

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
        st.subheader("Currently, data is available for the Angels and Rays only.")
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
