# mlb_agent_graph_refined.py
# --- Imports (combine necessary imports from previous agent script and ingestion script) ---
import pandas as pd
import json
from datetime import datetime, UTC
import requests
import logging
import time
from typing import Dict, List, Tuple, Any, Optional, TypedDict
import os
import re 
from ratelimit import limits, sleep_and_retry
from pydantic import BaseModel, Field
from google.cloud import secretmanager

# LangGraph and LangChain specific
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_google_vertexai import ChatVertexAI
from langchain.schema.output_parser import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Google Cloud specific
from google.cloud import bigquery, secretmanager
from google.api_core.exceptions import BadRequest, NotFound, PermissionDenied
import vertexai
from vertexai.language_models import TextEmbeddingModel, TextEmbeddingInput
from google.cloud import storage # Added GCS client
from vertexai.vision_models import Image, MultiModalEmbeddingModel
from tavily import TavilyClient
from image_embedding_pipeline2 import search_similar_images_sdk, execute_bq_query

# --- Configuration (Ensure these match ingestion script) ---
GCP_PROJECT_ID = "silver-455021"
GCP_LOCATION = "us-central1"
BQ_DATASET_ID = "mlb_rag_data_2024"
BQ_RAG_TABLE_ID = "rag_documents"      # For summaries & play snippets + embeddings
BQ_PLAYS_TABLE_ID = "plays"            # For structured play-by-play data
BQ_FULL_RAG_TABLE_ID = f"{GCP_PROJECT_ID}.{BQ_DATASET_ID}.{BQ_RAG_TABLE_ID}"
BQ_FULL_PLAYS_TABLE_ID = f"{GCP_PROJECT_ID}.{BQ_DATASET_ID}.{BQ_PLAYS_TABLE_ID}"
BQ_INDEX_NAME = "rag_docs_embedding_idx"

VERTEX_LLM_MODEL = "gemini-2.0-flash"
VERTEX_EMB_MODEL = "text-embedding-004"
EMBEDDING_TASK_TYPE = "RETRIEVAL_QUERY" # Use RETRIEVAL_QUERY for search queries
EMBEDDING_DIMENSIONALITY = 768
VERTEX_EMB_RPM = 1400 # Adjust
MLB_API_CALLS = 9
MLB_API_RATE_LIMIT = 60
PLAYER_METADATA_TABLE_ID = "mlb_player_metadata"
EMBEDDING_TABLE_ID = "mlb_image_embeddings_sdk"

BQ_DATASET_LOCATION = "US" # Actual location of your BQ dataset
VERTEX_MULTIMODAL_MODEL_NAME = "multimodalembedding@001"

# GCS Configuration
GCS_BUCKET_LOGOS = "mlb_logos"
GCS_PREFIX_LOGOS = "" # e.g., "logos/" if logos are in a subfolder
GCS_BUCKET_HEADSHOTS = "mlb-headshots"
GCS_PREFIX_HEADSHOTS = "headshots/" # e.g., "headshots/"

# --- Secret Manager Configuration ---
TAVILY_SECRET_ID = "TAVILY_SEARCH" # <-- REPLACE with your Secret ID in Secret Manager
TAVILY_SECRET_VERSION = "latest" #

# --- Logging and Clients (Initialize as before) ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Helper Function to Access Secret Manager ---
def access_secret_version(project_id: str, secret_id: str, version_id: str) -> Optional[str]:
    """Accesses a secret version from Google Secret Manager."""
    try:
        client = secretmanager.SecretManagerServiceClient()
        name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"
        response = client.access_secret_version(request={"name": name})
        payload = response.payload.data.decode("UTF-8")
        logger.info(f"Successfully accessed secret: {secret_id} (version: {version_id})")
        return payload
    except PermissionDenied:
        logger.error(f"Permission denied accessing secret: {secret_id}. Ensure the service account has 'Secret Manager Secret Accessor' role.")
        return None
    except NotFound:
         logger.error(f"Secret or version not found: projects/{project_id}/secrets/{secret_id}/versions/{version_id}")
         return None
    except Exception as e:
        logger.error(f"Error accessing secret {secret_id}: {e}", exc_info=True)
        return None

# --- Fetch Tavily API Key and Initialize Clients ---
tavily_api_key = access_secret_version(GCP_PROJECT_ID, TAVILY_SECRET_ID, TAVILY_SECRET_VERSION)
if tavily_api_key:
    os.environ["TAVILY_API_KEY"] = tavily_api_key
    logger.info("Tavily API key loaded into environment variable.")
    tavily = TavilyClient()
else:
    logger.warning("Tavily API key not found in Secret Manager or access failed. Web search node will not function.")
    # Decide how to handle this: exit, or let the node fail gracefully?
    # exit(1) # Option to exit if key is critical

try:
    # Simply call init(). If already initialized, it typically handles it gracefully.
    # If not initialized, this will set it up.
    vertexai.init(project=GCP_PROJECT_ID, location=GCP_LOCATION)
    logger.info(f"Ensured Vertex AI SDK is initialized for project {GCP_PROJECT_ID}, location {GCP_LOCATION}")

    # Initialize BQ Client
    bq_client = bigquery.Client(project=GCP_PROJECT_ID)
    logger.info(f"Initialized BigQuery client for project {GCP_PROJECT_ID}")

    # Initialize LangChain Model using ChatVertexAI
    model = ChatVertexAI(model_name=VERTEX_LLM_MODEL, project=GCP_PROJECT_ID, location=GCP_LOCATION, temperature=0.2)
    logger.info(f"Initialized LangChain ChatVertexAI model: {VERTEX_LLM_MODEL}")

    # Initialize model for structured output (optional, can use the same instance)
    structured_output_model = ChatVertexAI(model_name=VERTEX_LLM_MODEL, project=GCP_PROJECT_ID, location=GCP_LOCATION, temperature=0.0)
    logger.info(f"Initialized LangChain ChatVertexAI model for structured output: {VERTEX_LLM_MODEL}")

    # Initialize Embedding Model (direct SDK usage is fine here)
    emb_model = TextEmbeddingModel.from_pretrained(VERTEX_EMB_MODEL)
    logger.info(f"Initialized Vertex AI Embedding model: {VERTEX_EMB_MODEL}")

except Exception as e:
    logger.critical(f"Failed to initialize Google Cloud clients or LangChain Model: {e}", exc_info=True)
    raise RuntimeError("Critical initialization failed.") from e

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


# --- Agent State Definition (Added critique, revision tracking) ---
class AgentState(TypedDict):
    task: str
    game_pk: Optional[int]
    plan: str
    structured_data: Optional[Any] # Can be Dict or List[Dict] now
    narrative_context: Optional[List[str]]
    player_lookup_dict: Optional[Dict[int, str]] 
    image_search_queries: Optional[List[str]] # Queries derived *from* the final script
    retrieved_image_data: Optional[List[Dict[str, Any]]] # Results from image search
    draft: Optional[str]       # The current draft being worked on
    critique: Optional[str]    # Feedback from the reflection node
    generated_content: str     # Final output
    revision_number: int       # Start at 0, increment with each generation attempt
    max_revisions: int         # Max refinement loops
    error: Optional[str]

# --- BQ Query Functions (Keep from ingestion, slightly adapted for agent use) ---
def execute_bq_query(query: str) -> Optional[pd.DataFrame]:
    # (Same as in ingestion script)
    try:
        logger.info(f"Executing BQ Query: {query[:200]}...")
        query_job = bq_client.query(query)
        results = query_job.to_dataframe()
        logger.info(f"BQ Query returned {len(results)} rows.")
        return results
    except Exception as e:
        logger.error(f"Error executing BQ query: {query[:200]}... Error: {e}", exc_info=True)
        return None

def get_structured_game_metadata(game_pk: int) -> Optional[Dict[str, Any]]:
    """Fetches just the game summary doc metadata from BQ RAG table."""
    if not game_pk: return None
    query = f"""
    SELECT game_id, doc_type, metadata, last_updated
    FROM `{BQ_FULL_RAG_TABLE_ID}`
    WHERE game_id = {game_pk} AND doc_type = 'game_summary' LIMIT 1"""
    df = execute_bq_query(query)
    if df is not None and not df.empty:
        data_dict = df.iloc[0].to_dict()
        # Parse metadata JSON
        if isinstance(data_dict.get('metadata'), str):
            try: data_dict['metadata'] = json.loads(data_dict['metadata'])
            except json.JSONDecodeError: logger.warning(f"Could not parse metadata JSON (game {game_pk})")
        return data_dict['metadata'] # Return just the metadata dict
    return None

def get_structured_play_data(game_pk: int, play_filter_criteria: str = "1=1") -> Optional[List[Dict]]:
    """Fetches structured play data from the BQ Plays table, applying filters."""
    if not game_pk: return None
    # Basic safety check on criteria - more robust validation needed for production
    safe_criteria = re.sub(r"[^a-zA-Z0-9_=\s\<\>\'\.\-\(\),%]", "", play_filter_criteria) if play_filter_criteria else "1=1"

    query = f"""
    SELECT * EXCEPT(pitch_data, hit_data, runners_before, runners_after), -- Select core fields
           -- Select nested JSON data if needed, or process later
           pitch_data, hit_data, runners_after
    FROM `{BQ_FULL_PLAYS_TABLE_ID}`
    WHERE game_pk = {game_pk} AND {safe_criteria}
    ORDER BY play_index
    """
    df = execute_bq_query(query)
    if df is not None and not df.empty:
         # Convert DataFrame to list of dicts, parsing JSON strings back
        records = df.to_dict('records')
        for record in records:
            for col in ['pitch_data', 'hit_data', 'runners_after', 'runners_before']: # Add runners_before if stored
                 if col in record and isinstance(record[col], str):
                     try: record[col] = json.loads(record[col])
                     except json.JSONDecodeError: record[col] = None # Handle bad JSON
        logger.info(f"Retrieved {len(records)} structured plays for game {game_pk} with criteria '{safe_criteria}'")
        return records
    logger.warning(f"No structured plays found for game {game_pk} with criteria '{safe_criteria}'")
    return None

# Keep call_vertex_embedding from ingestion script
@sleep_and_retry
@limits(calls=VERTEX_EMB_RPM, period=60)
def call_vertex_embedding_agent(text_inputs: List[str]) -> List[Optional[List[float]]]:
    """Embedding specifically for agent queries/retrieval."""
    results = []
    batch_size = 200 # Adjust if needed
    try:
        all_embeddings = []
        for i in range(0, len(text_inputs), batch_size):
            batch = text_inputs[i:i + batch_size]
            # Use RETRIEVAL_QUERY type for the search query itself
            instances = [TextEmbeddingInput(text=text, task_type="RETRIEVAL_QUERY") for text in batch]
            kwargs = {"output_dimensionality": EMBEDDING_DIMENSIONALITY} # Only specify if not default
            embeddings_batch = emb_model.get_embeddings(instances, **kwargs)
            all_embeddings.extend([emb.values for emb in embeddings_batch])
            if len(text_inputs) > batch_size: time.sleep(1)
        return all_embeddings
    except Exception as e:
        logger.error(f"Error calling Vertex AI Embedding API: {e}", exc_info=True)
        return [None] * len(text_inputs)



# General query execution with type checking
def execute_filtered_query(table_name, column_name, filter_value, select_columns="*"):
    """
    Execute a query with type-safe filtering.
    """
    try:
        # Check if filter_value is a string with quotes
        if isinstance(filter_value, str) and (filter_value.startswith("'") or filter_value.startswith('"')):
            # Try to convert to integer by removing quotes
            try:
                filter_value_clean = int(filter_value.strip("'\""))
                # Use integer without quotes
                filter_clause = f"{column_name} = {filter_value_clean}"
            except ValueError:
                # Keep as string with quotes
                filter_clause = f"{column_name} = {filter_value}"
        else:
            # Assume it's already an integer or properly formatted
            filter_clause = f"{column_name} = {filter_value}"
        
        query = f"""
        SELECT {select_columns}
        FROM `{table_name}`
        WHERE {filter_clause}
        """
        
        logger.info(f"Executing filtered query on {table_name}")
        return execute_bq_query(query)
    except Exception as e:
        logger.error(f"Error executing filtered query: {e}", exc_info=True)
        return None


# --- Update the get_narrative_context_vector_search function ---
def get_narrative_context_vector_search(query_text: str, game_pk: Optional[int] = None, top_n: int = 5) -> List[str]:
    """
    Performs vector search, expecting a nested struct under 'base', and filters/sorts in Python.
    """
    if not query_text:
        logger.warning("Vector search query text is empty.")
        return []
    try:
        # 1. Get query embedding
        logger.info(f"Generating embedding for vector search query: '{query_text[:50]}...'")
        query_embedding_response = call_vertex_embedding_agent([query_text])
        if not query_embedding_response or not query_embedding_response[0]:
            logger.error("Failed to get embedding for vector search query.")
            return []
        query_embedding = query_embedding_response[0]
        query_embedding_str = f"[{', '.join(map(str, query_embedding))}]"

        # 2. Run VECTOR_SEARCH - Select base struct and distance
        initial_top_k = top_n * 10 + 30

        # ***** QUERY STRUCTURE v12 (Same SQL as v11, Python parsing changes) *****
        vector_search_query = f"""
        SELECT
            base,      -- Select the entire base row as a STRUCT/OBJECT
            distance   -- Select the distance calculated by VECTOR_SEARCH
        FROM
            VECTOR_SEARCH(
                TABLE `{BQ_FULL_RAG_TABLE_ID}`,
                'embedding',
                (SELECT {query_embedding_str} AS embedding),
                top_k => {initial_top_k},
                distance_type => 'COSINE'
            ) AS base -- Alias the results
        ORDER BY
            distance ASC
        LIMIT {initial_top_k}
        """
        # ***********************************************************************

        logger.info("Executing vector search selecting base struct and distance...")
        df_candidates = execute_bq_query(vector_search_query)

        if df_candidates is None or df_candidates.empty:
            logger.warning("Vector search returned no candidates.")
            return []

        # 3. Filter, Sort, and Limit results in Python using Pandas
        logger.info(f"Received {len(df_candidates)} candidates. Processing nested 'base' struct...")

        # Check if required top-level columns ('base', 'distance') exist
        if 'base' not in df_candidates.columns or 'distance' not in df_candidates.columns:
            logger.error(f"Required top-level columns ('base', 'distance') not found in results. Found: {df_candidates.columns.tolist()}. Cannot proceed.")
            return []

        # --- Extract data from the NESTED 'base' struct ---
        extracted_data = []
        logger.info(f"Inspecting first few rows of 'base' column (dtype: {df_candidates['base'].dtype}):") # Log dtype
        for index, row in df_candidates.head().iterrows(): # Log first few rows only
             logger.info(f"  Row {index}: Type = {type(row['base'])}, Value = {str(row['base'])[:500]}...") # Log type and truncated value
             if isinstance(row['base'], dict):
                   logger.info(f"  Row {index}: Keys in dict object = {list(row['base'].keys())}")

        for index, row in df_candidates.iterrows():
            nested_base_data = row['base'] # This is the outer dict {'query':..., 'base':..., 'distance':...} based on logs
            actual_base_struct = nested_base_data.get('base') # Try accessing the inner 'base' key
            distance = nested_base_data.get('distance') # Get distance from the outer dict

            # Check if actual_base_struct is a dict before accessing keys
            if isinstance(actual_base_struct, dict):
                extracted_data.append({
                    'doc_id': actual_base_struct.get('doc_id'),
                    'game_id': actual_base_struct.get('game_id'),
                    'content': actual_base_struct.get('content'),
                    'distance': distance # Use distance from outer dict
                })
            else:
                logger.warning(f"Row {index}: Inner 'base' data is not a dictionary (type: {type(actual_base_struct)}), skipping.")

        if not extracted_data:
             logger.warning("No valid data extracted after processing nested 'base' struct.")
             return []

        processed_df = pd.DataFrame(extracted_data)
        # --- Finished extraction ---

        # Ensure required columns exist in the new DataFrame
        required_cols = ['game_id', 'content', 'distance', 'doc_id']
        if not all(col in processed_df.columns for col in required_cols):
            logger.error(f"Required columns ({required_cols}) not found after extracting from struct. Found: {processed_df.columns.tolist()}. Cannot proceed.")
            return []

        # Filter by game_pk
        filtered_df = processed_df
        if game_pk:
            filtered_df['game_id'] = pd.to_numeric(filtered_df['game_id'], errors='coerce')
            filtered_df = filtered_df[filtered_df['game_id'] == game_pk].dropna(subset=['game_id'])
            logger.info(f"Filtered down to {len(filtered_df)} candidates for game_pk {game_pk}.")

        # Sort by distance and take top N
        final_df = filtered_df.sort_values(by='distance', ascending=True).head(top_n)

        if final_df.empty:
            logger.warning(f"No results remained after filtering/sorting for game_pk {game_pk}.")
            return []

        # 4. Extract the content
        results = final_df['content'].tolist()
        logger.info(f"Vector search with Python filter/sort returned {len(results)} final snippets.")
        return results

    except Exception as e:
        logger.error(f"Error during vector search (nested struct access attempt): {e}", exc_info=True)
        return []


# --- Refined Retriever Logic ---
# Define Pydantic models for structured LLM output for planning retrieval
class BQQuery(BaseModel):
    """A BigQuery SQL query designed to retrieve specific structured data."""
    query: str = Field(..., description="The SQL query to execute against BigQuery (use full table names like project.dataset.table). Filter by game_pk if relevant.")

class VectorSearch(BaseModel):
    """A query for semantic vector search."""
    query_text: str = Field(..., description="The natural language query to embed and search for in the vector store.")
    filter_by_game: bool = Field(True, description="Whether to restrict the search to the current game_pk.")

class RetrievalPlan(BaseModel):
    """Specifies which data retrieval methods to use."""
    structured_queries: Optional[List[BQQuery]] = Field(default_factory=list, description="List of BigQuery SQL queries for structured data.")
    vector_searches: Optional[List[VectorSearch]] = Field(default_factory=list, description="List of queries for vector search.")

RETRIEVER_PLANNER_PROMPT = """
You are a data retrieval expert for an MLB analysis system. Your goal is to decide HOW to fetch the data needed based on the user's task and the overall plan.

Available Data Sources:
1.  BQ Structured Metadata (`{rag_table}`): Game-level metadata. Key columns: `metadata`, `content`. Use `WHERE game_id = <pk> AND doc_type = 'game_summary'`.
2.  BQ Structured Plays (`{plays_table}`): Play-by-play data. Key columns: `inning`, `description`, `pitch_data`, `hit_data`, `rbi`, `event_type`. Use `WHERE game_pk = <pk>`.
3.  BQ Vector Search (`{rag_table}` column `embedding`): Narrative summaries & snippets. Filter by `game_id`. Key column: `content`.

User Request: {task}
Overall Plan: {plan}
Game ID (if applicable): {game_pk}

Determine the best retrieval actions. Output ONLY a JSON object containing two keys: "structured_queries" (a list of SQL query strings) and "vector_searches" (a list of strings for semantic search). If no queries of a type are needed, provide an empty list. Ensure the output is a single, valid JSON object.

JSON Output:
""" 

# --- Modify retrieve_data_node_refined to PARSE TEXT ---
def retrieve_data_node_refined(state: AgentState) -> Dict[str, Any]:
    """Generates queries as text, parses them, and executes retrieval."""
    logger.info("--- Refined Data Retrieval Node (Parsing Text) ---")
    task = state.get('task')
    plan = state.get('plan')
    game_pk = state.get('game_pk')

    if not plan: return {"error": "Plan is missing for retrieval."}

    prompt = RETRIEVER_PLANNER_PROMPT.format(
        task=task,
        plan=plan,
        game_pk=game_pk if game_pk else "Not Specified",
        rag_table=BQ_FULL_RAG_TABLE_ID,
        plays_table=BQ_FULL_PLAYS_TABLE_ID
    )

    retrieved_structured_data = []
    retrieved_narrative_context = []
    structured_queries_to_run = []
    vector_searches_to_run = []

    try:
        logger.info("Generating retrieval query plan (as text)...")
        # Use standard invoke, not with_structured_output
        response = model.invoke(prompt)
        llm_output_text = response.content
        logger.info(f"LLM Raw Output for Retrieval Plan:\n{llm_output_text}")

        # Parse the JSON output string
        try:
            # Clean potential markdown backticks
            json_match = re.search(r"```json\s*(\{.*?\})\s*```", llm_output_text, re.DOTALL | re.IGNORECASE)
            if json_match:
                json_string = json_match.group(1)
            else:
                json_string = llm_output_text.strip()

            retrieval_actions_dict = json.loads(json_string)
            # Extract lists, defaulting to empty list if key missing
            structured_queries_to_run = retrieval_actions_dict.get("structured_queries", [])
            vector_searches_to_run = retrieval_actions_dict.get("vector_searches", [])
            logger.info(f"Parsed {len(structured_queries_to_run)} structured queries and {len(vector_searches_to_run)} vector searches.")

            # Validate that queries are strings
            if not all(isinstance(q, str) for q in structured_queries_to_run):
                logger.error("Parsed structured_queries is not a list of strings.")
                structured_queries_to_run = [] # Reset on error
            if not all(isinstance(q, str) for q in vector_searches_to_run):
                logger.error("Parsed vector_searches is not a list of strings.")
                vector_searches_to_run = [] # Reset on error

        except (json.JSONDecodeError, AttributeError, KeyError, TypeError) as parse_error:
            logger.error(f"Failed to parse LLM output into retrieval actions: {parse_error}. Raw output: {llm_output_text}")
            # Fallback behavior
            if game_pk:
                 logger.warning("Falling back to basic game metadata retrieval.")
                 basic_meta = get_structured_game_metadata(game_pk)
                 if basic_meta: retrieved_structured_data.append(basic_meta)
            vector_searches_to_run = [task] # Default search based on original task


        # Execute Parsed Queries
        if structured_queries_to_run:
            logger.info("Executing structured queries...")
            for query_to_run in structured_queries_to_run:
                 if not isinstance(query_to_run, str) or not query_to_run.strip():
                      logger.warning("Skipping empty or invalid structured query.")
                      continue
                 # Simple safety check
                 if game_pk and f"{game_pk}" not in query_to_run:
                     logger.warning(f"Query '{query_to_run[:100]}...' might be missing game_pk filter.")
                 df_result = execute_bq_query(query_to_run)
                 if df_result is not None and not df_result.empty:
                     retrieved_structured_data.extend(df_result.to_dict('records'))
                 time.sleep(0.5)

        if vector_searches_to_run:
            logger.info("Executing vector searches...")
            for search_term in vector_searches_to_run:
                 if not isinstance(search_term, str) or not search_term.strip():
                      logger.warning("Skipping empty or invalid vector search term.")
                      continue
                 snippets = get_narrative_context_vector_search(search_term, game_pk) # Use the potentially fixed vector search function
                 retrieved_narrative_context.extend(snippets)
                 time.sleep(0.5)

    except Exception as e:
        logger.error(f"Error during refined data retrieval node execution: {e}", exc_info=True)
        # Fallback if needed
        if game_pk and not retrieved_structured_data:
             logger.warning("Falling back to basic game metadata retrieval after execution error.")
             basic_meta = get_structured_game_metadata(game_pk)
             if basic_meta: retrieved_structured_data.append(basic_meta)

    unique_narratives = list(dict.fromkeys(retrieved_narrative_context))
    logger.info(f"Retrieved {len(retrieved_structured_data)} structured data records.")
    logger.info(f"Retrieved {len(unique_narratives)} unique narrative snippets.")

    return {
        "structured_data": retrieved_structured_data if retrieved_structured_data else None,
        "narrative_context": unique_narratives if unique_narratives else None
    }

ANALYZE_SCRIPT_PROMPT = """
Analyze the following MLB script text. Identify all specific MLB teams and player full names mentioned.
For each team found, generate a search query string like "[Team Name] logo".
For each player found, generate a search query string like "[Player Full Name] headshot".
Output ONLY a JSON list of these search query strings. Return an empty list [] if no teams or players are found.

Example Input Script: "The Los Angeles Dodgers secured the win thanks to a great performance by Mookie Betts and Shohei Ohtani."
Example JSON Output:
["Los Angeles Dodgers logo", "Mookie Betts headshot", "Shohei Ohtani headshot"]

Script to Analyze:
{script}

JSON Output:
"""

def analyze_script_for_images_node(state: AgentState) -> Dict[str, Any]:
    """Analyzes the final script to generate image search queries."""
    logger.info("--- Analyze Script for Images Node ---")
    final_script = state.get('draft') # Use the final draft from the generation loop
    image_search_queries = []

    if not final_script:
        logger.warning("No final script found in state to analyze for images.")
        return {"image_search_queries": []} # Return empty list if no script

    prompt = ANALYZE_SCRIPT_PROMPT.format(script=final_script)

    try:
        logger.info("Generating image search queries based on script...")
        # Use the main 'model' instance, or 'structured_output_model' if preferred
        response = model.invoke(prompt)
        llm_output_text = response.content
        logger.info(f"LLM Raw Output for Image Queries:\n{llm_output_text}")

        # Parse JSON list
        try:
            json_match = re.search(r"```json\s*(\[.*?\])\s*```", llm_output_text, re.DOTALL | re.IGNORECASE)
            if json_match:
                json_string = json_match.group(1)
            else:
                # Try finding list directly if no markdown backticks
                list_match = re.search(r"(\[.*?\])", llm_output_text, re.DOTALL)
                json_string = list_match.group(1) if list_match else llm_output_text.strip()

            parsed_list = json.loads(json_string)
            if isinstance(parsed_list, list) and all(isinstance(item, str) for item in parsed_list):
                image_search_queries = parsed_list
                logger.info(f"Parsed {len(image_search_queries)} image search queries.")
            else:
                logger.error("LLM output for image queries was not a valid JSON list of strings.")
                image_search_queries = []

        except (json.JSONDecodeError, AttributeError, TypeError) as parse_error:
            logger.error(f"Failed to parse LLM output into image query list: {parse_error}. Raw output: {llm_output_text}")
            image_search_queries = [] # Default to empty list on error

    except Exception as e:
        logger.error(f"Error analyzing script for images: {e}", exc_info=True)
        image_search_queries = []

    return {"image_search_queries": image_search_queries}


# --- Helper function needed ---
def parse_player_name_from_query(search_term: str) -> Optional[str]:
    """Extracts player name from a search query like 'Player Name headshot'."""
    name = search_term.lower()
    # Remove common keywords
    name = name.replace(" headshot", "").replace("player ", "").strip()
    # Basic title casing (might need refinement for complex names)
    return name.title() if name else None

# --- Make sure storage_client is accessible or passed to the node ---
# It's initialized globally in the SDK script version, so it should be accessible.
# Add check_gcs_blob_exists helper if not already present
def check_gcs_blob_exists(bucket_name: str, blob_name: str) -> bool:
    """Checks if a blob exists in the GCS bucket."""
    try:
        bucket = storage_client.bucket(bucket_name) # Use the global client
        blob = bucket.blob(blob_name)
        exists = blob.exists()
        logger.debug(f"Checking existence for gs://{bucket_name}/{blob_name}: {exists}")
        return exists
    except Exception as e:
        logger.error(f"Error checking existence for blob {blob_name} in bucket {bucket_name}: {e}")
        return False # Assume it doesn't exist if check fails

# --- Revised retrieve_images_node ---
def retrieve_images_node(state: AgentState) -> Dict[str, Any]:
    """
    Retrieves images: Uses DIRECT LOOKUP for headshots based on Player ID,
    and VECTOR SEARCH for logos.
    """
    logger.info("--- Retrieve Images Node (Direct Headshot Lookup + Vector Logo Search) ---")
    image_searches_to_run = state.get("image_search_queries")
    player_lookup_dict = state.get("player_lookup_dict") or {}

    final_retrieved_images = [] # Store final selected images

    if not image_searches_to_run:
        logger.info("No image search queries provided.")
        return {"retrieved_image_data": []}

    # Create name -> ID lookup (case-insensitive)
    name_to_id_lookup = {name.lower(): pid for pid, name in player_lookup_dict.items()}

    processed_image_queries = set()
    logger.info(f"Processing {len(image_searches_to_run)} image search queries...")

    for search_term in image_searches_to_run:
        if not isinstance(search_term, str) or not search_term.strip() or search_term in processed_image_queries:
            continue
        processed_image_queries.add(search_term)

        selected_image_data = None # To hold the data for the selected image

        # --- Determine Query Type ---
        if "logo" in search_term.lower():
            # --- Logo Logic: Use Vector Search ---
            logger.info(f"Searching logo for '{search_term}' using Vector Search...")
            try:
                logo_candidates = search_similar_images_sdk(
                    query_text=search_term,
                    top_k=1
                )
                if logo_candidates:
                    selected_image_data = logo_candidates[0] # Take the top result
                    selected_image_data['search_term_origin'] = search_term
                    logger.info(f" -> Found logo: {selected_image_data.get('image_uri')}")
                else:
                    logger.warning(f"Vector search found no logo for '{search_term}'.")
            except Exception as search_err:
                logger.error(f"Error during logo vector search for '{search_term}': {search_err}", exc_info=True)
            # --- End Logo Logic ---

        elif "headshot" in search_term.lower() or "player" in search_term.lower():
            # --- Headshot Logic: Use Direct Lookup ---
            target_player_name = parse_player_name_from_query(search_term)

            if not target_player_name:
                logger.warning(f"Could not parse target player name from headshot query: '{search_term}'. Skipping.")
                continue # Skip if name cannot be parsed

            if not player_lookup_dict:
                 logger.warning(f"Player lookup empty, cannot perform direct lookup for '{target_player_name}'. Skipping.")
                 continue # Skip if lookup failed

            target_player_id = name_to_id_lookup.get(target_player_name.lower())

            if target_player_id:
                logger.info(f"Attempting direct lookup for '{target_player_name}' (ID: {target_player_id})...")
                # Construct the expected URI (ensure GCS bucket/prefix config vars are correct)
                # Assuming filename format headshot_{player_id}.jpg
                expected_blob_name = f"{GCS_PREFIX_HEADSHOTS}headshot_{target_player_id}.jpg"
                expected_uri = f"gs://{GCS_BUCKET_HEADSHOTS}/{expected_blob_name}"

                # Optional: Verify blob exists
                if check_gcs_blob_exists(GCS_BUCKET_HEADSHOTS, expected_blob_name):
                    logger.info(f"   --> Found and verified headshot via direct lookup: {expected_uri}")
                    # Create the result dictionary directly
                    selected_image_data = {
                        "image_uri": expected_uri,
                        "image_type": "headshot",
                        "entity_id": str(target_player_id), # Store ID as string
                        "entity_name": player_lookup_dict.get(target_player_id, target_player_name), # Get name from original dict
                        "distance": 0.0, # Indicate perfect match from lookup
                        "search_term_origin": search_term
                    }
                else:
                    logger.warning(f"   Direct lookup failed: Expected blob '{expected_blob_name}' not found in GCS for '{target_player_name}'.")
                    # Consider fallback? For now, we add nothing if direct lookup fails verification.

            else:
                # If the target player name wasn't in our metadata lookup
                logger.warning(f"Could not find player ID for name '{target_player_name}' in lookup table. Cannot retrieve headshot for '{search_term}'.")
            # --- End Headshot Logic ---
        else:
            logger.warning(f"Unrecognized image search type for query: '{search_term}'. Skipping.")

        # Add the selected image data (if any) to the final list
        if selected_image_data:
            final_retrieved_images.append(selected_image_data)

        time.sleep(0.2) # Small pause between processing each search term

    # Deduplicate final list (less likely needed with direct lookup but safe)
    unique_image_data = list({img['image_uri']: img for img in final_retrieved_images}.values())
    logger.info(f"Retrieved {len(unique_image_data)} unique image metadata records (Direct lookup for headshots).")

    return {"retrieved_image_data": unique_image_data}


def final_output_node(state: AgentState) -> Dict[str, Any]:
    """Copies the final draft and retrieved image data to final state fields."""
    logger.info("--- Final Output Node ---")
    final_draft = state.get("draft")
    image_data = state.get("retrieved_image_data") # Already a list or None

    output = {}
    if final_draft:
        logger.info("Setting final 'generated_content'.")
        output["generated_content"] = final_draft
    else:
        logger.warning("No draft available to set as final content.")
        output["error"] = state.get("error", "Final draft was missing.") # Preserve existing error

    # Pass through image data whether it was found or not
    output["retrieved_image_data"] = image_data
    logger.info(f"Final state includes {len(image_data) if image_data else 0} retrieved image records.")

    return output


# --- Updated REFLECTION_PROMPT ---
REFLECTION_PROMPT = """
You are a sharp, demanding MLB analyst and broadcast producer acting as a writing critic. Review the generated draft script based on the original request and plan.

Original Request: {task}
Plan: {plan}
Draft Script:
{draft}

Provide constructive criticism and specific, actionable recommendations for improvement. Be tough but fair. Focus on:
- **Accuracy:** Are ALL scores, stats (including exit velo, distance), player actions, and sequences correct based on typical game data? Point out ANY discrepancies or vagueness.
- **Compelling Narrative:** Does it have a strong hook? Does it build tension or excitement? Is the storytelling engaging for an MLB fan, or just a dry recitation of facts? Use stronger verbs, more vivid language.
- **Context:** Does it explain the *significance* of the game (e.g., playoff implications, rivalry, player milestones)? Does it explain *why* a stat (like exit velocity) is notable? Compare stats to averages if possible.
- **Impactful Play Descriptions:** Do they capture the moment? Add sensory details, crowd reaction context (even if inferred), explain the *impact* of the play beyond just the score change. Integrate pitch data (speed, type) if relevant and available.
- **Player Highlights:** Do they go beyond just listing stats? Connect performance to game narrative. Provide context on the player's role or typical performance.
- **Clarity and Conciseness:** Is it easy to follow? Is it too verbose or too brief for the target format (e.g., short video script)? Eliminate repetition. Be specific.
- **Data Integration:** Are stats used effectively to support the narrative, or just dropped in? Is there an opportunity to use *more* specific data points (pitch types, locations, defensive plays)?

If the draft is excellent and requires no changes (rare!), respond with "The draft looks excellent and fully addresses the request."
Otherwise, provide **specific, bulleted feedback** with clear examples of what needs improvement and *suggestions* for how to fix it. For instance, instead of saying "be more engaging," suggest *where* and *how* (e.g., "In the description of Higashioka's homer, add a sentence about the crowd's reaction or how it shifted the dugout energy.").
"""


RESEARCH_CRITIQUE_PROMPT = """
You are a research assistant. Based on the critique of the previous draft, generate specific search queries (max 3) to find information needed for the revision.

Critique:
{critique}

Focus on queries that will find facts, stats, context, or narrative examples to address the critique's points (e.g., specific player stats for that game, details about a key play mentioned, historical context). Use BQ table names `{rag_table}` and `{plays_table}` if suggesting SQL.
"""

def reflection_node(state: AgentState) -> Dict[str, str]:
    """Generates critique on the draft."""
    logger.info("--- Reflection Node ---")
    if not state.get('draft'): return {"error": "Draft missing for reflection."}
    prompt = REFLECTION_PROMPT.format(
        task=state['task'],
        plan=state['plan'],
        draft=state['draft']
    )
    try:
        response = model.invoke(prompt)
        critique = response.content
        logger.info(f"Critique: {critique}")
        return {"critique": critique}
    except Exception as e:
        logger.error(f"Error in reflection_node: {e}", exc_info=True)
        return {"error": f"Failed to generate critique: {e}", "critique": "Error generating critique."}

def research_critique_node(state: AgentState) -> Dict[str, Any]:
    """Generates research queries based on critique and fetches data."""
    logger.info("--- Research Critique Node ---")
    if not state.get('critique') or "excellent" in state.get('critique', '').lower():
         logger.info("Critique is positive or missing, skipping research.")
         # Return existing content so generate node can reuse it
         return {"content": state.get("content")} # Changed this: Pass previous content

    prompt = RESEARCH_CRITIQUE_PROMPT.format(
        critique=state['critique'],
        rag_table=BQ_FULL_RAG_TABLE_ID,
        plays_table=BQ_FULL_PLAYS_TABLE_ID
    )
    # Use the same retrieval logic, potentially informed by the critique
    # For simplicity, we'll just re-run a vector search based on the critique text
    # A more advanced system might parse the critique to generate specific BQ queries too
    new_narrative_context = []
    try:
        # ... (generate/execute research queries) ...
        logger.info("Performing vector search based on critique...")
        new_narrative_context = get_narrative_context_vector_search(state['critique'], state.get('game_pk'))

    except Exception as e:
         logger.error(f"Error generating/executing research queries from critique: {e}", exc_info=True)
         new_narrative_context = [] # Ensure it's a list even on error

    # ***** FIX TypeError *****
    # Explicitly handle if the existing context is None
    previous_narrative_context = state.get('narrative_context') or []
    combined_narrative = previous_narrative_context + new_narrative_context
    # *************************

    unique_combined_narrative = list(dict.fromkeys(combined_narrative)) # Deduplicate

    logger.info(f"Added {len(new_narrative_context)} new snippets based on critique. Total unique: {len(unique_combined_narrative)}")

    # Keep structured data from previous retrieval
    return {
        "narrative_context": unique_combined_narrative,
        "structured_data": state.get("structured_data") # Carry over structured data
    }
# --- Add this function definition with the other nodes ---

PLANNER_PROMPT_TEMPLATE = """
You are an expert MLB analyst and content planner. Your goal is to create a plan for fulfilling the user's request regarding an MLB game.
The available data includes structured game metadata (scores, teams, date, venue) and potentially narrative game summaries stored in a database.
Structured play-by-play data might also be queryable. Vector search can find relevant narrative context (like summaries or similar plays).

User Request: {task}
Game ID (if specified): {game_pk}

Based on the request, create a step-by-step plan focusing on:
1.  **Identifying Necessary Data:** Specify *what* data is needed (e.g., final score, specific player stats, summary text, similar historical plays).
2.  **Identifying Retrieval Methods:** Indicate *how* to get the data (e.g., 'BQ Query for metadata', 'Vector Search for narrative summary', 'BQ Query for play-by-play if needed').
3.  **Content Focus:** Briefly outline the key points the final content should cover to satisfy the user request, emphasizing storytelling using the data.

Output only the plan.

Plan:
"""

def planner_node(state: AgentState) -> Dict[str, Any]:
    """Generates a plan to fulfill the user's task."""
    logger.info("--- Planner Node ---")
    task = state.get('task')
    game_pk = state.get('game_pk') # game_pk might be None if not specified by user/task

    if not task:
        logger.error("Task is missing for planner.")
        # Return an error state or a default plan
        return {"error": "Task is missing.", "plan": "Error: Task missing."}

    prompt = PLANNER_PROMPT_TEMPLATE.format(
        task=task,
        game_pk=game_pk if game_pk else "Not Specified (Use latest or context if possible)"
    )
    try:
        # Ensure model is initialized correctly
        if 'model' not in globals() or not isinstance(model, ChatVertexAI):
             raise NameError("Global 'model' (ChatVertexAI) is not initialized.")

        response = model.invoke(prompt)
        plan = response.content
        logger.info(f"Generated Plan:\n{plan}")
        # Ensure plan is not empty
        if not plan or not plan.strip():
            logger.warning("Planner returned an empty plan, using default.")
            plan = "Default Plan: 1. Retrieve basic game info. 2. Generate summary."
            return {"plan": plan} # Return default plan but don't signal error yet

        return {"plan": plan}
    except Exception as e:
        logger.error(f"Error in planner_node: {e}", exc_info=True)
        # Return error and a default plan to potentially allow graceful failure
        return {"error": f"Failed to generate plan: {e}", "plan": "Default Plan due to error: Retrieve basic info."}

# Define Pydantic model for web search queries
class WebQueries(BaseModel):
    """Queries for web search."""
    queries: List[str] = Field(description="List of 1-3 targeted web search queries.")

# Prompt to generate web search queries based on critique
WEB_SEARCH_CRITIQUE_PROMPT = """
You are a research assistant specializing in finding external context for sports analysis.
Based on the critique of a generated MLB game recap, identify 1-3 key topics or questions raised in the critique that could be answered or enriched by a targeted web search.
Generate specific, concise search queries for the Tavily search engine to find this external information (e.g., player injury status before the game, historical significance of a milestone reached, recent team news impacting context, explanations of advanced stats mentioned).

Critique Provided:
{critique}

Generate a JSON object containing a single key "queries" which is a list of the search strings.
Example Output:
{{
  "queries": ["Nathan Eovaldi injury history 2024", "Rangers vs Red Sox rivalry significance", "what is wRC+ in baseball?"]
}}

JSON Output:
"""

@sleep_and_retry # Add rate limits if Tavily has them
@limits(calls=50, period=60) # Example Tavily limit - adjust as needed
def call_tavily_search(query: str, max_results: int = 2) -> List[str]:
    """Calls Tavily search API and returns a list of content snippets."""
    if not tavily:
        logger.warning("Tavily client not initialized, skipping web search.")
        return []
    try:
        logger.info(f"Performing Tavily search for: '{query}'")
        response = tavily.search(query=query, max_results=max_results, include_raw_content=False) # Adjust params as needed
        results = [r.get("content", "") for r in response.get("results", []) if r.get("content")]
        logger.info(f" -> Tavily returned {len(results)} results.")
        return results
    except Exception as e:
        logger.error(f"Error calling Tavily API for query '{query}': {e}", exc_info=True)
        return []

def web_search_critique_node(state: AgentState) -> Dict[str, Any]:
    """Generates web search queries from critique and executes them using Tavily."""
    logger.info("--- Web Search Critique Node ---")
    tavily = TavilyClient()
    critique = state.get("critique")
    current_narrative_context = state.get("narrative_context") or [] # Get existing context

    if not critique or "excellent" in critique.lower():
        logger.info("Critique positive or missing, skipping web search.")
        return {} # No change to narrative context

    if not tavily:
        logger.warning("Tavily client not available, skipping web search.")
        return {} # No change

    prompt = WEB_SEARCH_CRITIQUE_PROMPT.format(critique=critique)
    web_search_results = []
    web_queries = []

    try:
        logger.info("Generating web search queries based on critique...")
        # Use structured output model to get JSON query list
        response = structured_output_model.with_structured_output(WebQueries).invoke(prompt)
        web_queries = response.queries if response and response.queries else []
        logger.info(f"Generated {len(web_queries)} web search queries: {web_queries}")

    except Exception as e:
        logger.error(f"Failed to generate web search queries from critique: {e}", exc_info=True)
        # Optionally, could try a fallback query based on critique text itself

    # Execute web searches
    if web_queries:
        for query in web_queries:
            if query and isinstance(query, str):
                 results = call_tavily_search(query)
                 if results:
                     # Prepend context for clarity
                     web_search_results.extend([f"[Web Search Result for '{query}']: {res}" for res in results])
                 time.sleep(1) # Be polite to Tavily API
            else:
                 logger.warning(f"Skipping invalid web query: {query}")


    # Combine internal context and web search results
    if web_search_results:
         logger.info(f"Adding {len(web_search_results)} web snippets to narrative context.")
         # Add web results *after* internal context
         combined_context = current_narrative_context + web_search_results
         # Optional: Deduplicate or limit total context length if needed
         return {"narrative_context": combined_context}
    else:
         logger.info("No web search results obtained.")
         return {} # Return no changes if search yielded nothing


# --- Generate Node (Updated Prompt) ---
GENERATOR_PROMPT_REFINED_TEMPLATE = """
You are an expert MLB analyst and storyteller.
Original user request: "{task}"
Plan:
{plan}

You have already generated a draft, and received the following critique:
Critique:
{critique}

Based on the critique AND the available data(including internal data and external web search results), revise the draft or generate new content. Utilize all information below:

Structured Data:
```json
{structured_data_json}" \
"Narrative Context (Summaries, Play Snippets, Research based on Critique):
{narrative_context_str}

Instructions:

- Address the Critique: Explicitly incorporate the feedback from the critique, using web search results if they provide relevant context or facts.

- Synthesize ALL Data: Combine structured facts/stats with narrative context (internal and external). Clearly attribute information from web searches if necessary (e.g., "According to recent reports..." or implicitly use the fact).

- Deep Storytelling: Connect stats to game flow, explain significance, highlight key moments/matchups. Use specific details if available (pitch types, speeds, hit data).

- Fulfill Original Task: Ensure the final output clearly answers the user's request: "{task}".

Output the improved content in Markdown format.
"""



def generate_node_refined(state: AgentState) -> Dict[str, Any]:
  """Generates or revises content based on data, plan, and critique."""
  logger.info(f"--- Content Generation/Revision Node (Revision: {state.get('revision_number', 0)}) ---")
  task = state.get('task')
  plan = state.get('plan')
  critique = state.get('critique') # Might be None on first pass
  structured_data = state.get('structured_data')
  narrative_context = state.get('narrative_context', [])

  if state.get("error"): return {"error": state.get("error")}
  if not plan: return {"error": "Plan missing."}

  structured_data_json = json.dumps(structured_data, indent=2, default=str) if structured_data else "{}"
  narrative_context_str = "\n---\n".join(narrative_context) if narrative_context else "No narrative context available."

  # Use different prompts for first draft vs revision
  if critique and "excellent" not in critique.lower(): # Revision prompt
    prompt = GENERATOR_PROMPT_REFINED_TEMPLATE.format(
        task=task,
        plan=plan,
        critique=critique,
        structured_data_json=structured_data_json,
        narrative_context_str=narrative_context_str
    )
  else: # First draft prompt (or if critique was positive)
     # Use a slightly simpler prompt if no critique needs addressing
     prompt = f"""
        You are an expert MLB analyst and storyteller.
        User request: "{task}"
        Plan:
        {plan}

        Available Data:
        Structured:
        ```json
        {structured_data_json}
        ```
        Narrative Context:
        {narrative_context_str}

        Generate the content following the plan, using deep data storytelling by synthesizing structured stats and narrative context. Output in Markdown.
        """

  try:
    response = model.invoke(prompt)
    new_draft = response.content
    logger.info(f"Generated/Revised Draft (first 100 chars): {new_draft[:100]}...")
    # Increment revision number *after* successful generation
    current_revision = state.get('revision_number', 0)
    return {
        "draft": new_draft,
        "revision_number": current_revision + 1 # Increment happens here
        }
  except Exception as e:
    logger.error(f"Error in generate_node_refined: {e}", exc_info=True)
    return {"error": f"Failed to generate content: {e}", "draft": state.get("draft") or "Error generating draft."}

#--- Conditional Edge Logic ---
def should_continue(state: AgentState) -> str:
  """Determines whether to reflect or end the process."""
  logger.info("--- Should Continue Node ---")
  revision_number = state.get("revision_number", 1) # Generation increments it, so check > max
  max_revisions = state.get("max_revisions", 2)
  critique = state.get("critique", "")

  if state.get("error"):
    logger.error(f"Error detected: {state['error']}")
    return "END" # Or a specific error end node

  if revision_number > max_revisions:
    logger.info(f"Reached max revisions ({max_revisions}). Ending.")
    return "END"

  # Check if the critique node ran and gave positive feedback
  if critique and "excellent" in critique.lower():
     logger.info("Critique was positive. Ending.")
     return "END"

  logger.info(f"Revision {revision_number} <= {max_revisions}. Continuing to reflection.")
  return "reflect" # Continue the refinement loop

#--- Build the Graph ---
workflow = StateGraph(AgentState)

#Add nodes
workflow.add_node("planner", planner_node)
workflow.add_node("retrieve_data", retrieve_data_node_refined) # Use refined retriever
workflow.add_node("generate", generate_node_refined) # Use refined generator
workflow.add_node("reflect", reflection_node)
workflow.add_node("research_critique", research_critique_node)
workflow.add_node("analyze_script_for_images", analyze_script_for_images_node) # NEW
workflow.add_node("web_search_context", web_search_critique_node)
workflow.add_node("retrieve_images", retrieve_images_node) # NEW
workflow.add_node("final_output", final_output_node)

#Set entry point
workflow.set_entry_point("planner")

#Define edges
workflow.add_edge("planner", "retrieve_data")
workflow.add_edge("retrieve_data", "generate")

#Refinement loop
workflow.add_conditional_edges(
"generate", # Node to branch from
should_continue, # Function to decide the path
{
"reflect": "reflect", # If function returns "reflect", go to reflect node
"END": "web_search_context"  # <--- Transition to image analysis when text is final
}
)
workflow.add_edge("reflect", "research_critique")
workflow.add_edge("research_critique", "generate")

workflow.add_edge("web_search_context", "analyze_script_for_images") 

# Define edges for image retrieval path
workflow.add_edge("analyze_script_for_images", "retrieve_images")
workflow.add_edge("retrieve_images", "final_output") # Go to final output after image retrieval

# Define final end edge
workflow.add_edge("final_output", END)


memory = MemorySaver() # Optional: Add if chat history/memory is needed
app = workflow.compile(checkpointer=memory)
app = workflow.compile()


# --- Helper function (add this near the start of mlb_agent_graph_refined.py) ---
# Requires call_mlb_api to be defined in this script as well
@sleep_and_retry
@limits(calls=MLB_API_CALLS, period=MLB_API_RATE_LIMIT)
def call_mlb_api(url: str) -> Dict:
    # (Copy the definition from ingestion script here)
    try:
        response = requests.get(url, timeout=20)
        response.raise_for_status()
        if 'application/json' in response.headers.get('Content-Type', ''):
            return response.json()
        else:
            logger.warning(f"Non-JSON response: {url}. Status: {response.status_code}.")
            return {}
    except requests.exceptions.RequestException as e:
        logger.error(f"Error calling MLB API {url}: {e}")
        return {}

def get_latest_final_game_pk(team_id: int, season: int = 2024) -> Optional[int]:
    """Fetches the most recent *final* game ID for a specific team."""
    url = f'https://statsapi.mlb.com/api/v1/schedule?sportId=1&season={season}&teamId={team_id}&fields=dates,games,gamePk,officialDate,status,detailedState'
    logger.info(f"Fetching latest game for team {team_id}...")
    schedule_data = call_mlb_api(url)
    latest_game_pk = None
    latest_date = ""

    if schedule_data and 'dates' in schedule_data:
        all_final_games = []
        for date_entry in schedule_data.get('dates', []):
            for game in date_entry.get('games', []):
                 # Look specifically for 'Final' status
                 if game.get('status', {}).get('detailedState') == 'Final':
                    all_final_games.append({
                        'game_id': game.get('gamePk'),
                        'date': game.get('officialDate')
                    })

        if all_final_games:
            # Sort by date descending to get the most recent
            all_final_games.sort(key=lambda x: x['date'], reverse=True)
            latest_game_pk = all_final_games[0]['game_id']
            latest_date = all_final_games[0]['date']
            logger.info(f"Found latest final game for team {team_id}: PK {latest_game_pk} on {latest_date}")

    if not latest_game_pk:
         logger.warning(f"No recent *final* game ID found for team {team_id}, season {season}.")

    return latest_game_pk


# --- Updated Example Usage (at the end of mlb_agent_graph_refined.py) ---
if __name__ == "__main__":
    logger.info("\n--- Running Refined Agent Graph ---")

    # --- Dynamic Game PK ---
    # Choose a default team ID to find the latest game for (e.g., Rangers = 140)
    default_team_id_for_latest = 144
    latest_game_pk = get_latest_final_game_pk(default_team_id_for_latest)

    if not latest_game_pk:
        logger.error(f"Could not find the latest game PK for team {default_team_id_for_latest}. Exiting example.")
        exit() # Or handle differently, maybe fallback to a known good PK

    logger.info(f"Using latest game PK found: {latest_game_pk}")
    # -----------------------

    # Task can now reference the "latest game" implicitly or explicitly
    # task = f"Analyze the key moments and pitching duel in the latest game involving team ID {default_team_id_for_latest} (Game PK: {latest_game_pk})."
    task = f"Provide a detailed recap of game {latest_game_pk}, highlighting impactful plays and player performances." # Example task

    max_loops = 2

    # --- *** MOVED: Load Player Metadata BEFORE Initial State *** ---
    logger.info("Loading player metadata into memory for lookups...")
    player_lookup_dict = {} # Initialize as empty dict
    try:
        # Ensure necessary tables exist first (optional, but good practice)
        # setup_player_metadata_table() # Uncomment if needed

        player_lookup_query = f"SELECT player_id, player_name FROM `{GCP_PROJECT_ID}.{BQ_DATASET_ID}.{PLAYER_METADATA_TABLE_ID}`"
        player_results_df = execute_bq_query(player_lookup_query) # Get DataFrame

        if player_results_df is not None and not player_results_df.empty:
            # Use .iterrows() to build the dictionary
            player_lookup_dict = {int(row['player_id']): row['player_name'] for index, row in player_results_df.iterrows() if pd.notna(row['player_id'])}
            logger.info(f"Loaded {len(player_lookup_dict)} player names into lookup dictionary.")
        else:
            logger.warning("Player metadata query failed or returned no results. Lookup dictionary is empty.")
    except Exception as meta_err:
         logger.error(f"Failed to load player metadata: {meta_err}. Proceeding with empty lookup.", exc_info=True)
         player_lookup_dict = {} # Ensure it's an empty dict on error
    # --- *** END MOVED BLOCK *** ---

    initial_state = {
        "task": task,
        "game_pk": latest_game_pk, # Use the dynamically found PK
        "max_revisions": max_loops,
        "revision_number": 0,
        "plan": None,
        "structured_data": None,
        "narrative_context": [],
        "player_lookup_dict": player_lookup_dict,
        "image_search_queries":None, 
        "retrieved_image_data":None,        
        "draft": None,
        "critique": None,
        "generated_content": None,
        "error": None,
    }

    logger.info(f"\nExecuting graph for Task: {task}")

    # Use invoke to get the final state directly
    try:
        # Make sure the graph is compiled correctly before invoking
        if 'app' not in globals():
             # Re-compile if needed (ensure workflow definition is complete above)
             logger.warning("Re-compiling graph 'app'...")
             app = workflow.compile()

        # ***** INCREASED RECURSION LIMIT *****
        # Set a higher, fixed limit or a more generous calculation
        # recursion_limit = max_loops * 4 + 5 # Generous calculation
        recursion_limit = 50 # Or just a fixed higher number
        # *************************************

        final_state = app.invoke(initial_state, {"recursion_limit": recursion_limit})
        if final_state.get("error"):
             print("\n--- Execution Failed ---")
             print(f"Error: {final_state['error']}")
        elif final_state.get("retrieved_image_data"): # Check 'draft' as it holds the last generated content
             print("\n--- Final Generated Content ---")
             print(json.dumps(final_state["retrieved_image_data"], indent=2, default=str))
             print(final_state["draft"])
        else:
            print("\n--- Execution Finished, but no final draft found in state. Check logs. ---")
            print("Final state snapshot:", {k: v for k,v in final_state.items() if k != 'structured_data' and k != 'narrative_context'}) # Avoid printing huge data

    except Exception as e:
         logger.error(f"Error invoking graph: {e}", exc_info=True)
         print(f"\n--- Graph Invocation Error ---")
         print(f"An exception occurred: {e}")