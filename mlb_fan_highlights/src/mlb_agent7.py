# mlb_agent_graph_refined.py
# --- Imports (combine necessary imports from previous agent script and ingestion script) ---
import io
import random
import shutil
from PIL import Image as PilImage
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
from pydub import AudioSegment 

# LangGraph and LangChain specific
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_google_vertexai import ChatVertexAI
from langchain.schema.output_parser import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from google import genai

# Google Cloud specific
from google.cloud import bigquery, secretmanager
from google.api_core.exceptions import BadRequest, NotFound, PermissionDenied
from google.api_core import exceptions as google_exceptions
import vertexai
from vertexai.language_models import TextEmbeddingModel, TextEmbeddingInput
from google.cloud import storage # Added GCS client
from vertexai.vision_models import Image, MultiModalEmbeddingModel
from tavily import TavilyClient
from image_embedding_pipeline2 import search_similar_images_sdk, execute_bq_query
#import google.generativeai as genai # Use alias genai for clarity
#from google.generativeai.types import GenerateVideosConfig, Image as GenAiImage 
from google.genai import types as genai_types # Alias types to avoid conflict
from google.cloud import texttospeech_v1 as texttospeech # Ensure this import is present


# --- Add near other Vertex AI imports ---
from vertexai.preview.vision_models import ImageGenerationModel
import google.cloud.storage # Ensure storage client is used for saving generated images too


# Imagen Generation Parameters
IMAGE_GENERATION_SEED = None # Set to None initially, Imagen 3 might handle random seed internally better without conflicts. Can add back if needed.
IMAGE_GENERATION_WATERMARK = False # Set to False if using Seed, or if you just don't want it.
IMAGE_GENERATION_NEGATIVE_PROMPT = "text, words, letters, blurry, low quality, cartoonish, illustration, drawing, sketch, unrealistic, watermark, signature, writing"
IMAGE_GENERATION_ASPECT_RATIO = "16:9" # Common video aspect ratio
IMAGE_GENERATION_NUMBER_PER_PROMPT = 1
VIDEO_GENERATION_PERSON_ALLOW = "allow_adult"

# --- Add near other model configurations ---
VERTEX_IMAGEN_MODEL = "imagen-3.0-generate-002" # From the advertising notebook
VERTEX_VEO_MODEL_ID = "veo-2.0-generate-001" # From the retail notebook (for potential future video)
GCS_BUCKET_GENERATED_ASSETS = "mlb_generated_assets" # NEW: Define a bucket for generated images/videos
# Ensure this bucket exists in your GCP project!
VIDEO_GENERATION_PROMPT = "Subtle camera pan, slow motion effect, cinematic lighting." # Default prompt for image-to-video
VIDEO_GENERATION_ASPECT_RATIO = "16:9" # Should match image aspect ratio
VIDEO_GENERATION_PERSON_ALLOW = "allow_adult" # Or "allow_all" / "block_adult" etc.
VIDEO_GENERATION_SLEEP_SECONDS = 15 # Sleep between starting video generations
VIDEO_POLLING_INTERVAL_SECONDS = 30 # How often to check if video operation is done
VIDEO_GENERATION_QUOTA_SLEEP_SECONDS = 90 # Sleep after quota error starting generation
VIDEO_GENERATION_ERROR_SLEEP_SECONDS = 20 # Sleep after other errors starting generation
MAX_IMAGES_TO_ANIMATE = 3 # Limit how many images get turned into videos (per run)
GCS_BUCKET_GENERATED_VIDEOS = "mlb_generated_videos" # NEW: Define a bucket/prefix for generated videos
# Ensure GCS_BUCKET_GENERATED_VIDEOS bucket exists in your GCP project!
VIDEO_DURATION_SECONDS = 7 # Example duration from docs
VIDEO_ENHANCE_PROMPT = True # Let Veo enhance the prompt (useful since prompts were image-focused)
MAX_PROMPTS_TO_ANIMATE = 3 # Limit how many prompts get turned into videos

GCS_VIDEO_OUTPUT_PREFIX = "generated/videos/"

# You might want to make these configurable
IMAGE_GENERATION_SLEEP_SECONDS = 35 # Sleep between successful calls
IMAGE_GENERATION_ERROR_SLEEP_SECONDS = 15 # Sleep after a general error
IMAGE_GENERATION_QUOTA_SLEEP_SECONDS = 70 # Longer sleep after hitting quota
CLOUDFLARE_FALLBACK_SLEEP_SECONDS = 5     # Sleep after a Cloudflare attempt

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

GOOGLE_GENAI_USE_VERTEXAI=True

# --- Secret Manager Configuration ---
TAVILY_SECRET_ID = "TAVILY_SEARCH" # <-- REPLACE with your Secret ID in Secret Manager
TAVILY_SECRET_VERSION = "latest" #

# --- Add near other configurations ---
CLOUDFLARE_ACCOUNT_ID_SECRET = "cloudflare-account-id" # Your Secret Manager ID
CLOUDFLARE_API_TOKEN_SECRET = "cloudflare-api-token" # Your Secret Manager ID
CLOUDFLARE_FALLBACK_MODEL = "@cf/bytedance/stable-diffusion-xl-lightning" # Or choose another
CLOUDFLARE_API_ENDPOINT_TEMPLATE = "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model_id}"

# Team configurations (shortened for brevity)
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


# --- Store credentials globally after fetching ---
cloudflare_account_id = None
cloudflare_api_token = None

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

# --- Add this section after initializing Vertex/BQ/etc. clients ---

logger.info("Attempting to load Cloudflare credentials from Secret Manager...")
try:
    # Use the existing access_secret_version function
    cloudflare_account_id = access_secret_version(GCP_PROJECT_ID, CLOUDFLARE_ACCOUNT_ID_SECRET, "latest")
    cloudflare_api_token = access_secret_version(GCP_PROJECT_ID, CLOUDFLARE_API_TOKEN_SECRET, "latest")

    if not cloudflare_account_id or not cloudflare_api_token:
        logger.warning("Cloudflare Account ID or API Token not found in Secret Manager. Fallback generation will be disabled.")
        cloudflare_account_id = None # Ensure they are None if fetch failed
        cloudflare_api_token = None
    else:
        logger.info("Cloudflare credentials loaded successfully. Fallback generation enabled.")

except Exception as e:
    logger.error(f"Failed to load Cloudflare credentials: {e}. Fallback generation disabled.", exc_info=True)
    cloudflare_account_id = None
    cloudflare_api_token = None

# --- End Cloudflare credential loading ---

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


# --- Add near other client initializations ---
try:
    # ... existing clients ...
    imagen_model = ImageGenerationModel.from_pretrained(VERTEX_IMAGEN_MODEL)
    logger.info(f"Initialized Vertex AI Imagen model: {VERTEX_IMAGEN_MODEL}")
    # veo_model = None # Initialize Veo model here if/when implementing video generation
    # logger.info(f"Initialized Vertex AI Veo model: {VERTEX_VEO_MODEL}") # If using Veo

    # Ensure storage client is initialized (it should be already)
    if 'storage_client' not in globals():
        storage_client = storage.Client(project=GCP_PROJECT_ID)
        logger.info("Initialized Google Cloud Storage client.")

except Exception as e:
    logger.critical(f"Failed to initialize Imagen/Veo/Storage clients: {e}", exc_info=True)
    # Decide how to handle - potentially exit or disable generation features
    imagen_model = None
    # veo_model = None

# --- Near other client initializations ---
genai_client = None # Initialize as None
try:
    # Initialize vertexai SDK first (already done)
    # vertexai.init(...)

    # Initialize google-genai client specifically for Veo, pointing to Vertex
    # Uses the same project/location as vertexai.init
    genai_client = genai.Client(vertexai=True, project=GCP_PROJECT_ID, location=GCP_LOCATION)
    logger.info(f"Initialized google-genai Client for Veo (Project: {GCP_PROJECT_ID}, Location: {GCP_LOCATION})")

    # Initialize Imagen model using vertexai SDK (already done)
    # imagen_model = ImageGenerationModel.from_pretrained(...)

    # ... other clients (BQ, GCS, Multimodal, Tavily) ...

except Exception as e:
    logger.error(f"Failed to initialize google-genai Client for Veo: {e}", exc_info=True)
    # Veo generation will be skipped if client is None

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
    generated_content: str  
    all_image_assets: Optional[List[Dict[str, Any]]] # Combined static and generated images
    all_video_assets: Optional[List[Dict[str, Any]]] # Combined generated videos   # Final output
    revision_number: int       # Start at 0, increment with each generation attempt
    max_revisions: int         # Max refinement loops
    # --- NEW: Visual Generation Loop ---
    visual_generation_prompts: Optional[List[str]] # Prompts for Imagen/Veo
    generated_visual_assets: Optional[List[Dict[str, Any]]] # Results from Imagen/Veo
    visual_critique: Optional[str] # Critique specifically for generated visuals
    visual_revision_number: int # Counter for visual generation loop
    max_visual_revisions: int   # Max loops for visual generation (e.g., 2)
    generated_video_assets: Optional[List[Dict[str, Any]]]
    generated_audio_uri: Optional[str] # GCS URI for the final audio
    error: Optional[str]

def load_image_bytes_from_gcs(gcs_uri: str) -> Optional[Tuple[bytes, str]]:
    """Loads image bytes and determines mime type from a GCS URI."""
    global storage_client
    if not gcs_uri or not gcs_uri.startswith("gs://"):
        logger.error(f"Invalid GCS URI provided: {gcs_uri}")
        return None
    if 'storage_client' not in globals() or storage_client is None:
        logger.error("GCS storage client not initialized. Cannot load image bytes.")
        return None

    try:
        bucket_name, blob_name = gcs_uri.replace("gs://", "").split("/", 1)
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        if not blob.exists():
            logger.error(f"Image blob not found at GCS URI: {gcs_uri}")
            return None

        image_bytes = blob.download_as_bytes()

        # Determine mime type (simple check based on extension)
        if blob_name.lower().endswith(".png"):
            mime_type = "image/png"
        elif blob_name.lower().endswith(".jpg") or blob_name.lower().endswith(".jpeg"):
            mime_type = "image/jpeg"
        else:
            # Attempt to guess or default
            content_type = blob.content_type
            if content_type and content_type.startswith("image/"):
                 mime_type = content_type
            else:
                 logger.warning(f"Could not determine specific image mime type for {gcs_uri}, defaulting to image/png.")
                 mime_type = "image/png" # Default guess

        logger.info(f"Successfully loaded {len(image_bytes)} bytes from {gcs_uri} (Mime: {mime_type})")
        return image_bytes, mime_type

    except Exception as e:
        logger.error(f"Error loading image bytes from GCS {gcs_uri}: {e}", exc_info=True)
        return None

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


VISUAL_PROMPT_ANALYSIS_PROMPT = """
You are an assistant director analyzing an MLB game script to plan visual shots, specifically for the Imagen 3 text-to-image model which has filters against specific names.
Read the script carefully. Your primary goal is to identify 3-5 key moments, scenes, or actions that need a generated visual.

**Critical Instructions for Imagen Compatibility:**
1.  **NO Player Names:** Absolutely **DO NOT** use any real player names (e.g., "Ohtani", "Judge"). Use generic descriptions like "an MLB player", "the batter", "the pitcher", "a fielder", "the runner".
2.  **NO Team Names:** Absolutely **DO NOT** use specific MLB team names (e.g., "Dodgers", "Yankees").
3.  **Uniform Descriptions:**
    *   If the script explicitly states a player is on the **home team**, describe their uniform generically as such (e.g., "a player in a white home uniform", "the batter in a home jersey"). Assume home uniforms are primarily white or light grey unless the script specifies otherwise.
    *   If the script explicitly states a player is on the **away team**, describe their uniform generically as such (e.g., "a player in a colored away uniform", "the pitcher in a gray away jersey"). Assume away uniforms are colored or dark grey unless the script specifies otherwise.
    *   If the script provides **specific color details** for a uniform (e.g., "wearing blue and orange"), use those details.
    *   If the script **does not specify** home/away or give color details for the relevant player/action, use a neutral description like "an MLB player's uniform".

**Prompt Generation Guidelines:**
*   For actions (like a home run, double play, strikeout), break them down into 2-4 distinct visual prompts representing the sequence (e.g., swing, ball flight, player running, tag/catch, celebration).
*   For descriptive moments (e.g., stadium shot, manager looking tense), generate a single detailed prompt.
*   Focus on creating descriptive prompts suitable for Imagen 3. Emphasize action, emotion, setting, and relevant details like uniform descriptions based on the rules above.

**Example Input Script:** "The Dodgers' star player crushed a high fastball while playing at home, sending it deep into the right-field stands for a 3-run homer! The crowd went wild as he rounded the bases."

**Example JSON Output (Reflecting New Rules):**
[
  "An MLB batter in a white home uniform swinging a baseball bat powerfully, follow-through motion, intense focus, stadium background during daytime.",
  "Baseball soaring high in the air against a blue sky, heading towards the right-field seats of a packed baseball stadium.",
  "An MLB player in a white home uniform jogging around third base, smiling, pointing upwards, during a baseball game.",
  "Wide shot of a baseball stadium crowd cheering ecstatically, fans on their feet, after a home run."
]

**Script to Analyze:**
{script}

**Output Format:** Output ONLY a JSON list of prompt strings. Keep the list concise (maximum 5-7 prompts total unless the script is exceptionally long and detailed). Ensure every prompt adheres strictly to the NO Player Name and NO Team Name rules, and uses uniform descriptions as specified.
**JSON Output:**
"""

def analyze_script_for_visual_prompts_node(state: AgentState) -> Dict[str, Any]:
    """Analyzes the final script to generate prompts for Imagen/Veo."""
    logger.info("--- Analyze Script for Visual Generation Prompts Node ---")
    final_script = state.get('draft')

    if not final_script:
        logger.warning("No final script found to analyze for visual prompts.")
        return {"visual_generation_prompts": []}

    if not imagen_model: # Check if model initialized
         logger.error("Imagen model not available, cannot generate visual prompts.")
         return {"visual_generation_prompts": [], "error": "Imagen model not initialized."}


    prompt = VISUAL_PROMPT_ANALYSIS_PROMPT.format(script=final_script)
    visual_prompts = []

    try:
        logger.info("Generating visual generation prompts based on script...")
        # Use the standard model for this analysis task
        response = model.invoke(prompt)
        llm_output_text = response.content
        logger.debug(f"LLM Raw Output for Visual Prompts:\n{llm_output_text}")

        # Parse JSON list (similar to analyze_script_for_images_node)
        try:
            # Basic parsing, add more robust handling if needed
            json_match = re.search(r"```json\s*(\[.*?\])\s*```", llm_output_text, re.DOTALL | re.IGNORECASE)
            if json_match:
                json_string = json_match.group(1)
            else:
                list_match = re.search(r"(\[.*?\])", llm_output_text, re.DOTALL)
                json_string = list_match.group(1) if list_match else llm_output_text.strip()

            parsed_list = json.loads(json_string)
            if isinstance(parsed_list, list) and all(isinstance(item, str) for item in parsed_list):
                visual_prompts = parsed_list
                logger.info(f"Parsed {len(visual_prompts)} visual generation prompts.")
            else:
                logger.error("LLM output for visual prompts was not a valid JSON list of strings.")
                visual_prompts = []

        except (json.JSONDecodeError, AttributeError, TypeError) as parse_error:
            logger.error(f"Failed to parse LLM output into visual prompt list: {parse_error}. Raw output: {llm_output_text}")
            visual_prompts = []

    except Exception as e:
        logger.error(f"Error analyzing script for visual prompts: {e}", exc_info=True)
        visual_prompts = []

    # Initialize visual revision number here before the first generation attempt
    return {
        "visual_generation_prompts": visual_prompts,
        "visual_revision_number": 0, # Start counter at 0 before first generation
        "generated_visual_assets": [] # Ensure it's an empty list initially
        }



# Helper function to save PIL image to GCS (adapt if needed)
# Helper function to save PIL image to GCS (adapt if needed)
def save_image_to_gcs(image: PilImage.Image, bucket_name: str, blob_name: str) -> Optional[str]: # Changed PIL_Image to Image
    """Saves a PIL Image object to GCS and returns the gs:// URI."""
    try:
        # Ensure storage_client is accessible (initialized globally or passed)
        global storage_client
        if 'storage_client' not in globals() or storage_client is None:
             logger.error("GCS storage client not initialized. Cannot save image.")
             return None

        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        # Save PIL image to a temporary in-memory buffer
        buffer = io.BytesIO()
        image.save(buffer, format='PNG') # Or JPEG, ensure 'image' is the PIL Image object
        buffer.seek(0)

        blob.upload_from_file(buffer, content_type='image/png') # Or image/jpeg
        gcs_uri = f"gs://{bucket_name}/{blob_name}"
        logger.info(f"Successfully saved generated image to {gcs_uri}")
        return gcs_uri
    except Exception as e:
        logger.error(f"Error saving image to GCS gs://{bucket_name}/{blob_name}: {e}", exc_info=True)
        return None

def generate_video_clips_node(state: AgentState) -> Dict[str, Any]:
    """Generates short video clips directly from text prompts using Veo."""
    node_start_time = time.time()
    logger.info("--- Generate Video Clips Node (Text-to-Video) ---")
    # --- Use the prompts generated for images as input for videos ---
    video_prompts = state.get("visual_generation_prompts") or []
    current_videos = state.get("generated_video_assets") or []

    global genai_client # Use the specific genai client for Veo
    if not genai_client:
        logger.warning("google-genai client not initialized, skipping video generation.")
        return {"generated_video_assets": current_videos}

    if not video_prompts:
        logger.info("No visual generation prompts found to generate videos from.")
        return {"generated_video_assets": current_videos}

    # --- Select prompts to turn into videos ---
    # Simple selection: Take the first MAX_PROMPTS_TO_ANIMATE
    prompts_to_animate = video_prompts[:MAX_PROMPTS_TO_ANIMATE]
    logger.info(f"Selected {len(prompts_to_animate)} prompts to generate videos for (max: {MAX_PROMPTS_TO_ANIMATE}).")

    if not prompts_to_animate:
        logger.info("No prompts selected for video generation.")
        return {"generated_video_assets": current_videos}

    # --- Generate videos from prompts ---
    newly_generated_videos = []
    game_pk = state.get("game_pk", "unknown_game")
    timestamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S")

    for i, prompt_text in enumerate(prompts_to_animate):
        logger.info(f"Attempting text-to-video generation {i+1}/{len(prompts_to_animate)} for prompt: '{prompt_text[:80]}...'")
        operation_lro = None # Variable to hold the LRO object
        operation_name_str = None # Variable to hold the operation name string
        # --- Prepare Veo API call ---
        try:
            # Create unique output URI for the video
            prompt_slug = re.sub(r'\W+', '_', prompt_text[:30]).strip('_') # Slug from prompt
            video_blob_name = f"{GCS_VIDEO_OUTPUT_PREFIX}game_{game_pk}/vid_{prompt_slug}_{timestamp}_{i:02d}.mp4"
            output_video_gcs_uri = f"gs://{GCS_BUCKET_GENERATED_VIDEOS}/{video_blob_name}"

            # Prepare config using genai module directly
            veo_config = genai_types.GenerateVideosConfig(
                output_gcs_uri=output_video_gcs_uri,
                aspect_ratio=VIDEO_GENERATION_ASPECT_RATIO,
                number_of_videos=1,
                duration_seconds=VIDEO_DURATION_SECONDS, # Set duration
                person_generation=VIDEO_GENERATION_PERSON_ALLOW,
                enhance_prompt=VIDEO_ENHANCE_PROMPT, # Use enhance prompt
            )

            # --- Call Veo API (starts long-running operation) ---
            logger.info(f"Starting Veo text-to-video operation...")
            operation_lro = genai_client.models.generate_videos(
                model=VERTEX_VEO_MODEL_ID,
                prompt=prompt_text, # Use the text prompt directly
                # REMOVED: image=veo_image_input,
                config=veo_config,
            )
            # Get the name just for logging, handle potential absence
            operation_name_for_log = getattr(operation_lro, 'name', 'UNKNOWN_OPERATION_NAME')
            logger.info(f"Veo operation started: {operation_name_for_log}. Polling for completion...")

            # --- Poll the operation using the OBJECT ---
            polling_start_time = time.time()
            current_op_state = operation_lro # Start with the initial object

            while not current_op_state.done:
                time.sleep(VIDEO_POLLING_INTERVAL_SECONDS)
                try:
                    # --- FIX: Pass the object itself to get() to refresh ---
                    current_op_state = genai_client.operations.get(current_op_state) # Pass the object
                    elapsed = time.time() - polling_start_time
                    # Use the name we got initially for consistent logging
                    logger.debug(f"Polling Veo operation {operation_name_for_log} (elapsed: {elapsed:.0f}s)... Done: {current_op_state.done}")
                except Exception as poll_err:
                     logger.error(f"Error refreshing Veo operation {operation_name_for_log}: {poll_err}. Stopping polling for this video.")
                     current_op_state = None # Mark as failed if refresh fails
                     break # Exit polling loop

            polling_duration = time.time() - polling_start_time

            # --- Process result ---
            # Check the final state of the operation object
            if current_op_state and current_op_state.done and not current_op_state.error:
                if current_op_state.response:
                    video_uri = current_op_state.result.generated_videos[0].video.uri
                    logger.info(f"Veo text-to-video successful for prompt '{prompt_text[:80]}...'. Operation: {operation_name_for_log}, URI: {video_uri} (Polling took {polling_duration:.2f}s)")
                    newly_generated_videos.append({
                         "source_prompt": prompt_text,
                         "video_uri": video_uri,
                         "model_used": VERTEX_VEO_MODEL_ID,
                         "type": "generated_video"
                     })
                else:
                    logger.warning(f"Veo operation {operation_name_for_log} finished successfully but has no response/result object.")
            elif current_op_state and current_op_state.error:
                 error_details = current_op_state.error
                 logger.error(f"Veo operation {operation_name_for_log} finished WITH FAILED status for prompt '{prompt_text[:80]}...'. Error: {error_details}. (Polling took {polling_duration:.2f}s)")
            elif current_op_state is None: # Polling loop broke
                logger.error(f"Veo operation FAILED during polling for prompt '{prompt_text[:80]}...'.")
            else: # Final state is unexpected
                 done_status = getattr(current_op_state, 'done', 'N/A')
                 error_status = getattr(current_op_state, 'error', 'N/A')
                 logger.warning(f"Veo operation {operation_name_for_log} finished in unexpected state (Done: {done_status}, Error: {error_status}) for prompt '{prompt_text[:80]}...'")


            logger.debug(f"Sleeping {VIDEO_GENERATION_SLEEP_SECONDS}s before starting next video generation.")
            time.sleep(VIDEO_GENERATION_SLEEP_SECONDS)

        except google_exceptions.ResourceExhausted as quota_error:
            logger.error(f"Veo Quota Exceeded starting text-to-video generation for prompt '{prompt_text[:80]}...': {quota_error}. Sleeping {VIDEO_GENERATION_QUOTA_SLEEP_SECONDS}s.")
            time.sleep(VIDEO_GENERATION_QUOTA_SLEEP_SECONDS)
            # Skip to next prompt

        except google_exceptions.FailedPrecondition as fp_error:
             # Catch potential permission/allowlist errors specifically
             logger.error(f"Veo Failed Precondition starting text-to-video for prompt '{prompt_text[:80]}...': {fp_error}. Check project permissions/allowlists.")
             time.sleep(VIDEO_GENERATION_ERROR_SLEEP_SECONDS)

        except Exception as e:
            logger.error(f"Unexpected error during Veo text-to-video setup/start for prompt '{prompt_text[:80]}...': {e}", exc_info=True)
            logger.debug(f"Sleeping {VIDEO_GENERATION_ERROR_SLEEP_SECONDS}s after error.")
            time.sleep(VIDEO_GENERATION_ERROR_SLEEP_SECONDS)
            # Skip to next prompt

    # --- Node Summary ---
    all_video_assets = current_videos + newly_generated_videos
    node_duration = time.time() - node_start_time
    logger.info(f"--- Video Clips Node (Text-to-Video) Summary ---")
    logger.info(f"  Duration: {node_duration:.2f}s")
    logger.info(f"  Prompts attempted: {len(prompts_to_animate)}")
    logger.info(f"  Videos successfully generated: {len(newly_generated_videos)}")
    logger.info(f"  Total video assets now: {len(all_video_assets)}")

    return {"generated_video_assets": all_video_assets}

import requests # Add near other imports
from io import BytesIO # Already needed for saving

# --- Add this helper function definition ---

def generate_image_cloudflare(prompt: str, width: int = 768, height: int = 768, num_steps: int = 20) -> Optional[bytes]:
    """
    Generates an image using Cloudflare Worker AI (Stable Diffusion fallback).

    Args:
        prompt: The text prompt for image generation.
        width: Image width.
        height: Image height.
        num_steps: Number of diffusion steps.

    Returns:
        Image content as bytes if successful, otherwise None.
    """
    global cloudflare_account_id, cloudflare_api_token # Access the global credentials

    if not cloudflare_account_id or not cloudflare_api_token:
        logger.warning("Cloudflare credentials not available, skipping fallback generation.")
        return None

    url = CLOUDFLARE_API_ENDPOINT_TEMPLATE.format(
        account_id=cloudflare_account_id,
        model_id=CLOUDFLARE_FALLBACK_MODEL
    )

    headers = {
        "Authorization": f"Bearer {cloudflare_api_token}",
        "Content-Type": "application/json"
    }

    data = {
        "prompt": prompt,
        "width": width,
        "height": height,
        "num_steps": num_steps,
        # "guidance": 7.5 # Optional, model default might be fine
    }

    logger.info(f"Attempting fallback generation with Cloudflare ({CLOUDFLARE_FALLBACK_MODEL}) for prompt: '{prompt[:80]}...'")

    try:
        response = requests.post(url, headers=headers, json=data, timeout=60) # Add timeout

        if response.status_code == 200:
            logger.info("Cloudflare fallback generation successful.")
            return response.content
        else:
            logger.error(f"Cloudflare fallback failed. Status: {response.status_code}, Response: {response.text[:200]}...")
            return None

    except requests.exceptions.RequestException as e:
        logger.error(f"Error during Cloudflare API request: {e}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Unexpected error during Cloudflare fallback generation: {e}", exc_info=True)
        return None

# --- End Cloudflare helper function ---
# (Make sure imports are present: time, random, io, requests, google_exceptions, Image from PIL)


def generate_visuals_node(state: AgentState) -> Dict[str, Any]:
    """
    Generates images using the configured Imagen model with appropriate sleep,
    error handling, and Cloudflare fallback.
    """
    node_start_time = time.time()
    logger.info(f"--- Generate Visuals Node (Revision: {state.get('visual_revision_number', 0)}) ---")
    prompts = state.get("visual_generation_prompts")
    current_assets = state.get("generated_visual_assets") or []

    if not prompts:
        logger.warning("No visual generation prompts provided.")
        return {"generated_visual_assets": current_assets} # Return existing assets

    # --- Check if primary Imagen model is available ---
    global imagen_model
    primary_model_available = False
    if 'imagen_model' in globals() and imagen_model is not None:
        primary_model_available = True
    else:
        logger.error(f"Primary Imagen model ({VERTEX_IMAGEN_MODEL}) not initialized.")
        # Check if fallback is possible
        if not cloudflare_account_id or not cloudflare_api_token:
             logger.error("Imagen model unavailable AND Cloudflare fallback disabled. Cannot generate visuals.")
             # Return error to potentially stop the loop gracefully
             return {
                 "generated_visual_assets": current_assets,
                 "error": "Primary image generation model unavailable and fallback disabled."
             }
        else:
             logger.warning("Imagen model unavailable, will attempt Cloudflare fallback ONLY.")

    newly_generated_assets = []
    game_pk = state.get("game_pk", "unknown_game")
    timestamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S")

    # --- Prepare Generation Parameters from Config ---
    generation_params = {
        "number_of_images": IMAGE_GENERATION_NUMBER_PER_PROMPT,
        "aspect_ratio": IMAGE_GENERATION_ASPECT_RATIO,
        "add_watermark": IMAGE_GENERATION_WATERMARK,
    }
    # Conditionally add seed and negative prompt
    if IMAGE_GENERATION_SEED is not None and not IMAGE_GENERATION_WATERMARK:
        generation_params["seed"] = IMAGE_GENERATION_SEED # Or use random.randint(0, 2**31 - 1) per call if SEED is None
    elif IMAGE_GENERATION_WATERMARK:
        logger.warning("Watermark is enabled, seed parameter will be ignored by Imagen.")

    if IMAGE_GENERATION_NEGATIVE_PROMPT:
        generation_params["negative_prompt"] = IMAGE_GENERATION_NEGATIVE_PROMPT

    successful_generations = 0
    failed_generations = 0
    fallback_attempts = 0
    fallback_successes = 0

    # --- Loop through prompts ---
    num_prompts_to_process = len(prompts)
    for i, prompt_text in enumerate(prompts):
        # Skip if already generated in a previous visual revision loop for this task run
        # Note: This check assumes prompts don't change between visual revisions.
        # A more robust check might involve comparing image URIs if prompts could be regenerated.
        if any(asset.get("prompt_origin") == prompt_text for asset in current_assets):
             logger.info(f"Skipping prompt already generated in a previous revision: '{prompt_text[:80]}...'")
             continue

        logger.info(f"Processing visual prompt {i+1}/{num_prompts_to_process}: '{prompt_text[:80]}...'")

        imagen_succeeded = False
        gcs_uri = None
        model_used = None
        pil_image_object = None
        prompt_start_time = time.time()

        # --- Attempt 1: Configured Imagen Model ---
        if primary_model_available:
            try:
                # Use a random seed per image if main SEED is None and watermark is off
                current_seed = generation_params.get("seed")
                if current_seed is None and not IMAGE_GENERATION_WATERMARK:
                    current_seed = random.randint(1, 2**31 - 1) # Generate random seed
                    logger.debug(f"Using random seed for this prompt: {current_seed}")
                else:
                     current_seed = generation_params.get("seed") # Use configured seed or None if watermark on

                logger.info(f"Attempting Imagen generation ({VERTEX_IMAGEN_MODEL})...")
                response = imagen_model.generate_images(
                    prompt=prompt_text,
                    seed=current_seed, # Pass the determined seed
                    **{k:v for k,v in generation_params.items() if k != 'seed'} # Pass other params except seed
                )
                prompt_duration = time.time() - prompt_start_time
                logger.info(f"Imagen API call completed in {prompt_duration:.2f} seconds.")

                # --- Check Response ---
                # Updated check based on SDK structure
                if response and response.images: # Check if response and images list exist
                     # Directly access the PIL image via the intended property if available
                     # Assuming response.images[0] gives an object with a ._pil_image attribute
                     # Or potentially just response.images[0] IS the PIL image? Need to confirm SDK structure.
                     # Let's assume the ._pil_image attribute for now based on previous context.
                     img_candidate = response.images[0]._pil_image # Access internal attribute

                     if isinstance(img_candidate, PilImage.Image): # Check if it's a valid PIL Image
                        pil_image_object = img_candidate
                        model_used = VERTEX_IMAGEN_MODEL
                        imagen_succeeded = True
                        successful_generations += 1
                        logger.info(f" --> Imagen generation successful.")
                     else:
                        logger.warning(f"Imagen response for prompt {i+1} did not contain a valid PIL Image object.")
                        failed_generations += 1
                else:
                    safety_ratings = getattr(response, 'safety_ratings', 'N/A')
                    logger.warning(f"Imagen returned no image(s) for prompt {i+1}. Possible safety filter? Ratings: {safety_ratings}")
                    failed_generations += 1 # Count as failed if no image returned

                # --- Sleep AFTER Imagen Success or known failure (but NOT Quota Error) ---
                # Use the configured long sleep time
                logger.info(f"Sleeping {IMAGE_GENERATION_SLEEP_SECONDS}s after successful Imagen attempt.")
                time.sleep(IMAGE_GENERATION_SLEEP_SECONDS)


            except google_exceptions.ResourceExhausted as quota_error:
                prompt_duration = time.time() - prompt_start_time
                logger.error(f"Imagen Quota Exceeded for prompt {i+1} after {prompt_duration:.2f}s: {quota_error}. Sleeping {IMAGE_GENERATION_QUOTA_SLEEP_SECONDS}s.")
                failed_generations += 1
                time.sleep(IMAGE_GENERATION_QUOTA_SLEEP_SECONDS)
                # DO NOT attempt fallback immediately after quota error. Skip to next prompt.
                continue

            except google_exceptions.InvalidArgument as arg_error:
                prompt_duration = time.time() - prompt_start_time
                logger.error(f"Imagen Invalid Argument for prompt {i+1} after {prompt_duration:.2f}s: {arg_error}. Will attempt fallback.")
                failed_generations += 1
                time.sleep(IMAGE_GENERATION_ERROR_SLEEP_SECONDS) # Shorter sleep before fallback

            except Exception as e:
                prompt_duration = time.time() - prompt_start_time
                logger.error(f"Unexpected Imagen Error for prompt {i+1} after {prompt_duration:.2f}s: {e}", exc_info=True)
                logger.warning("Will attempt fallback due to unexpected Imagen error.")
                failed_generations += 1
                time.sleep(IMAGE_GENERATION_ERROR_SLEEP_SECONDS) # Shorter sleep before fallback

        else:
            # If primary model wasn't loaded, log it and proceed to fallback
            logger.info(f"Primary Imagen model not loaded, proceeding directly to fallback for prompt {i+1}.")


        # --- Attempt 2: Cloudflare Fallback (if Imagen failed for reasons OTHER than quota) ---
        if not imagen_succeeded and primary_model_available: # Only fallback if Imagen was tried and failed (non-quota)
             fallback_attempts += 1
             logger.info("Attempting Cloudflare fallback...")
             cloudflare_image_bytes = generate_image_cloudflare(prompt_text) # Use helper

             if cloudflare_image_bytes:
                 try:
                     fallback_image = PilImage.open(BytesIO(cloudflare_image_bytes))
                     if isinstance(fallback_image, PilImage.Image): # Verify conversion
                         pil_image_object = fallback_image
                         model_used = CLOUDFLARE_FALLBACK_MODEL # Note the fallback model used
                         fallback_successes += 1
                         # Overwrite failed generation count since fallback worked
                         failed_generations -= 1 # Decrement failure count
                         successful_generations +=1 # Increment success count
                         logger.info(f" --> Cloudflare fallback successful.")
                     else:
                         logger.warning(f"Could not convert Cloudflare response to valid PIL Image for prompt {i+1}.")
                         # Failed generations count remains incremented from Imagen failure

                 except Exception as conversion_err:
                     logger.error(f"Error converting Cloudflare image bytes to PIL Image: {conversion_err}", exc_info=True)
                     # Failed generations count remains incremented
             else:
                 logger.warning(f" --> Cloudflare fallback also failed for prompt {i+1}.")
                 # Failed generations count remains incremented

             # Sleep after fallback attempt (success or fail)
             logger.debug(f"Sleeping {CLOUDFLARE_FALLBACK_SLEEP_SECONDS}s after fallback attempt.")
             time.sleep(CLOUDFLARE_FALLBACK_SLEEP_SECONDS)

        elif not primary_model_available: # Case where Imagen wasn't available from the start
             fallback_attempts += 1
             logger.info("Attempting Cloudflare fallback (primary model was unavailable)...")
             # (Duplicate Cloudflare logic for clarity - could be refactored)
             cloudflare_image_bytes = generate_image_cloudflare(prompt_text)
             if cloudflare_image_bytes:
                 try:
                     fallback_image = PilImage.open(BytesIO(cloudflare_image_bytes))
                     if isinstance(fallback_image, PilImage.Image):
                         pil_image_object = fallback_image
                         model_used = CLOUDFLARE_FALLBACK_MODEL
                         fallback_successes += 1
                         successful_generations +=1 # Count as success
                         logger.info(f" --> Cloudflare fallback successful.")
                     else:
                         logger.warning(f"Could not convert Cloudflare response to valid PIL Image.")
                         failed_generations += 1 # Count as failure if conversion fails
                 except Exception as conversion_err:
                     logger.error(f"Error converting Cloudflare image bytes to PIL Image: {conversion_err}", exc_info=True)
                     failed_generations += 1
             else:
                 logger.warning(f" --> Cloudflare fallback also failed.")
                 failed_generations += 1

             logger.debug(f"Sleeping {CLOUDFLARE_FALLBACK_SLEEP_SECONDS}s after fallback attempt.")
             time.sleep(CLOUDFLARE_FALLBACK_SLEEP_SECONDS)


        # --- Save the resulting image (if generation succeeded with either model) ---
        if pil_image_object and model_used:
            try:
                rev_num = state.get('visual_revision_number', 0)
                # Use a more descriptive blob name
                prompt_slug = re.sub(r'\W+', '_', prompt_text[:30]).strip('_') # Basic slug from prompt
                blob_name = f"generated/game_{game_pk}/img_{timestamp}_rev{rev_num}_{i+1:02d}_{prompt_slug}.jpg" # Use JPEG

                # Use the helper function to save
                gcs_uri = save_image_to_gcs(pil_image_object, GCS_BUCKET_GENERATED_ASSETS, blob_name)

                if gcs_uri:
                    newly_generated_assets.append({
                        "prompt_origin": prompt_text,
                        "image_uri": gcs_uri,
                        "model_used": model_used,
                        "type": "generated_image",
                        "revision": rev_num # Store revision number with the asset
                    })
                    # Success already counted when pil_image_object was set
                else:
                     logger.warning(f"Failed to save generated image to GCS for prompt {i+1} (model: {model_used}).")
                     # If saving fails, we should probably decrement success and increment failure
                     successful_generations -= 1
                     failed_generations += 1

            except Exception as save_err:
                 logger.error(f"Failed to save image from {model_used} for prompt {i+1} to GCS: {save_err}", exc_info=True)
                 successful_generations -= 1
                 failed_generations += 1
        # No else needed here, failure count is handled within the try/except blocks


    # --- Node Summary ---
    all_generated_assets = current_assets + newly_generated_assets
    current_revision = state.get('visual_revision_number', 0) # Revision number before incrementing
    node_duration = time.time() - node_start_time
    logger.info(f"--- Visuals Node (Rev: {current_revision}) Summary ---")
    logger.info(f"  Duration: {node_duration:.2f}s")
    logger.info(f"  Prompts Processed: {num_prompts_to_process}")
    logger.info(f"  Successful Generations: {successful_generations}")
    logger.info(f"  Failed Generations: {failed_generations}")
    logger.info(f"  Fallback Attempts: {fallback_attempts}")
    logger.info(f"  Fallback Successes: {fallback_successes}")
    logger.info(f"  Total Assets Now: {len(all_generated_assets)}")

    output_state = {
        "generated_visual_assets": all_generated_assets,
         # Increment revision number *before* returning from the node
        "visual_revision_number": current_revision + 1
    }
    # Optional: Check if ALL prompts failed despite retries/fallbacks
    # if successful_generations == 0 and num_prompts_to_process > 0:
    #    logger.error("All visual generation prompts failed.")
    #    output_state["error"] = "Failed to generate any visual assets for the provided prompts."

    return output_state


# Remember to have the corrected save_image_to_gcs function available
# def save_image_to_gcs(image: Image.Image, bucket_name: str, blob_name: str) -> Optional[str]: ...
VISUAL_CRITIQUE_PROMPT = """
You are a demanding visual producer reviewing generated images for an MLB highlight segment.
The script is:
--- SCRIPT ---
{script}
--- END SCRIPT ---

The following images were generated based on specific prompts:
--- GENERATED IMAGES (by original prompt) ---
{generated_images_summary}
--- END GENERATED IMAGES ---

Critique the generated images:
1.  **Coverage:** Do the images adequately cover the key moments described in the script? Are there obvious gaps?
2.  **Relevance:** Does each image reasonably match its intended prompt and the corresponding script moment?
3.  **Quality/Action:** Are the images clear? Do they convey the intended action or mood (even if static)? (Note: Don't expect perfect photorealism, but judge relevance).
4.  **Suggestions:** If improvements are needed, suggest *specific* new prompts or modifications. What's missing? (e.g., "Need an image of the runner sliding into second base", "Generate a shot focusing on the pitcher's frustrated reaction after the homer").

If the current set of images is sufficient and adequately covers the script, respond ONLY with "Visuals look sufficient."
Otherwise, provide concise, bullet-point feedback focusing on gaps and specific prompts needed for the *next* generation round.
"""

def critique_visuals_node(state: AgentState) -> Dict[str, Any]:
    """Critiques the generated visual assets."""
    logger.info("--- Critique Visuals Node ---")
    script = state.get("draft")
    generated_assets = state.get("generated_visual_assets")

    if not script or not generated_assets:
        logger.warning("Script or generated visuals missing, cannot critique.")
        # If visuals are missing, maybe request generation? Or just mark as sufficient?
        # Let's assume if assets are missing, we can't proceed meaningfully.
        return {"visual_critique": "Visuals look sufficient."} # Default to sufficient if no assets generated

    # Create a summary for the prompt
    summary_lines = []
    for asset in generated_assets:
         summary_lines.append(f"- Prompt: '{asset.get('prompt_origin', 'N/A')[:80]}...' -> URI: {asset.get('image_uri', 'N/A')}")
    generated_images_summary = "\n".join(summary_lines)

    prompt = VISUAL_CRITIQUE_PROMPT.format(
        script=script,
        generated_images_summary=generated_images_summary
    )

    try:
         logger.info("Generating critique for visual assets...")
         # Use the standard model
         response = model.invoke(prompt)
         critique = response.content.strip()
         logger.info(f"Visual Critique: {critique}")
         return {"visual_critique": critique}

    except Exception as e:
         logger.error(f"Error generating visual critique: {e}", exc_info=True)
         # Default to sufficient on error to avoid infinite loop
         return {"visual_critique": "Visuals look sufficient.", "error": "Failed to generate visual critique."}

def should_continue_visuals(state: AgentState) -> str:
    """Determines whether to regenerate visuals or finalize."""
    logger.info("--- Should Continue Visuals Node ---")
    visual_revision_number = state.get("visual_revision_number", 0)
    max_visual_revisions = state.get("max_visual_revisions", 2) # Use state value if set
    visual_critique = state.get("visual_critique", "")
    error = state.get("error")

    if error:
        logger.error(f"Error detected in visual loop: {error}")
        return "aggregate_final_output" # Proceed to end on error

    if visual_revision_number >= max_visual_revisions:
        logger.info(f"Reached max visual revisions ({max_visual_revisions}). Aggregating output.")
        return "aggregate_final_output"

    if "sufficient" in visual_critique.lower():
        logger.info("Visual critique is positive. Aggregating output.")
        return "aggregate_final_output"

    # If critique is not sufficient and haven't reached max revisions, generate again
    logger.info(f"Visual critique suggests more needed (Revision {visual_revision_number}/{max_visual_revisions}). Generating again.")
    # **Crucial:** Decide if you re-use original prompts or need new ones.
    # For now, let's assume we re-run generation. A better approach needs
    # a node to parse critique and *create* new prompts.
    # We might need to clear the critique here so we don't loop infinitely if generation doesn't improve.
    # state['visual_critique'] = None # Potential strategy, needs testing
    return "generate_visuals" # Loop back to generate node

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
You are a sharp, demanding MLB analyst and broadcast producer acting as a writing critic. Review the generated **two-host dialogue script** based on the original request and plan.

Original Request: {task}
Plan: {plan}
Dialogue Draft:
{draft}

Provide constructive criticism and specific, actionable recommendations for improvement. Be tough but fair. Focus on:
- **Dialogue Flow & Engagement:** Does the conversation sound natural? Is the back-and-forth engaging? Do the hosts have distinct enough 'voices' or roles (e.g., analyst vs. color commentator)? Is it just question/answer or a real discussion?
- **Accuracy:** Are ALL scores, stats (including exit velo, distance), player actions, and sequences correct based on typical game data? Point out ANY discrepancies or vagueness.
- **Compelling Narrative:** Does the *conversation* have a strong hook? Does it build tension or excitement? Is the storytelling engaging for an MLB fan, or just a dry recitation of facts passed between hosts?
- **Context:** Does the dialogue explain the *significance* of the game (e.g., playoff implications, rivalry, player milestones)? Does it explain *why* a stat (like exit velocity) is notable? Compare stats to averages if possible within the conversation.
- **Impactful Play Descriptions:** Do the hosts' descriptions capture the moment? Add sensory details, crowd reaction context (even if inferred), explain the *impact* of the play beyond just the score change. Integrate pitch data (speed, type) if relevant and available *within the dialogue*.
- **Player Highlights:** Do they go beyond just listing stats? Connect performance to game narrative *through the hosts' discussion*. Provide context on the player's role or typical performance.
- **Clarity and Conciseness:** Is the dialogue easy to follow? Is it too verbose or too brief? Eliminate repetition between hosts.
- **Data Integration:** Are stats used effectively to support the conversation, or just dropped in? Is there an opportunity to use *more* specific data points?

If the draft is excellent and requires no changes (rare!), respond with "The dialogue draft looks excellent and fully addresses the request."
Otherwise, provide **specific, bulleted feedback** with clear examples of what needs improvement and *suggestions* for how to fix it within the dialogue format.
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

def save_audio_to_gcs(audio_bytes: bytes, bucket_name: str, blob_name: str) -> Optional[str]:
    """Saves audio bytes to GCS and returns the gs:// URI."""
    try:
        global storage_client
        if 'storage_client' not in globals() or storage_client is None:
             logger.error("GCS storage client not initialized. Cannot save audio.")
             return None

        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        # Upload directly from bytes
        blob.upload_from_string(audio_bytes, content_type='audio/mpeg') # For MP3

        gcs_uri = f"gs://{bucket_name}/{blob_name}"
        logger.info(f"Successfully saved generated audio to {gcs_uri}")
        return gcs_uri
    except Exception as e:
        logger.error(f"Error saving audio to GCS gs://{bucket_name}/{blob_name}: {e}", exc_info=True)
        return None



# --- Make sure these imports are present ---
from google.cloud import texttospeech
from pydub import AudioSegment # Requires pydub library (pip install pydub)
import io                      # Requires ffmpeg to be installed on the system
import os
import tempfile # For creating temporary files safely

# --- Configuration for Audio Node (Add Alt Voice) ---
TTS_VOICE_NAME = "en-US-Chirp3-HD-Puck"           # Voice for Host 1 (even lines: 0, 2, 4...)
TTS_VOICE_NAME_ALT = "en-US-Chirp3-HD-Aoede"      # Voice for Host 2 (odd lines: 1, 3, 5...) # Example WaveNet voice
# Or use other Chirp voices:
# TTS_VOICE_NAME = "en-US-Chirp3-HD-Puck"
# TTS_VOICE_NAME_ALT = "en-US-Chirp3-HD-Aoede"

AUDIO_ENCODING = texttospeech.AudioEncoding.MP3
GCS_BUCKET_GENERATED_AUDIO = "mlb_generated_audio" # Ensure this bucket exists
AUDIO_SILENCE_MS = 350 # Silence between speakers in milliseconds

# Helper function (should already exist)
# def save_audio_to_gcs(audio_bytes: bytes, bucket_name: str, blob_name: str) -> Optional[str]: ...

# --- NEW Multi-Speaker generate_audio_node ---
def generate_audio_node(state: AgentState) -> Dict[str, Any]:
    """Generates multi-speaker audio from the final dialogue script using TTS and pydub."""
    node_start_time = time.time()
    logger.info("--- Generate Multi-Speaker Audio Node ---")
    final_script = state.get("generated_content")
    game_pk = state.get("game_pk", "unknown_game")
    existing_audio_uri = state.get("generated_audio_uri")

    # Check prerequisites
    try:
        # Check if pydub is installed (optional but helpful for user feedback)
        from pydub import AudioSegment
    except ImportError:
        logger.error("pydub library not found. Please install it (`pip install pydub`) and ensure ffmpeg is installed.")
        return {"error": "Missing pydub library or ffmpeg dependency."}

    # If audio already exists
    if existing_audio_uri and state.get("word_timestamps"):
        logger.info(f"Audio already generated: {existing_audio_uri}")
        return {"generated_audio_uri": existing_audio_uri, "word_timestamps": state.get("word_timestamps")}

    if not final_script:
        logger.warning("No final script ('generated_content') found to synthesize audio.")
        return {"error": "Missing final script for audio generation."}

    # Parse the dialogue script
    dialogue_lines = [line.strip() for line in final_script.splitlines() if line.strip()]
    if not dialogue_lines:
        logger.warning("Final script contains no valid dialogue lines.")
        return {"error": "Script has no content for audio generation."}

    # --- Initialize TTS Client ---
    try:
        # Use v1 client specifically, often better documented for timestamps
        tts_client = texttospeech.TextToSpeechClient()
        logger.info("TextToSpeechClient (v1) initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize TextToSpeechClient (v1): {e}", exc_info=True)
        return {"error": "Failed to initialize TTS client."}

    combined_audio_segments = [] # Store pydub segments directly
    all_word_timestamps = []
    current_time_offset_s = AUDIO_SILENCE_MS / 1000.0 # Start with initial silence

    temp_dir = tempfile.mkdtemp()
    logger.info(f"Created temporary directory for audio parts: {temp_dir}")

    try:
        logger.info(f"Generating {len(dialogue_lines)} audio segments with timestamps...")
        for count, line in enumerate(dialogue_lines):
            if count % 2 == 0:
                current_voice_name = TTS_VOICE_NAME
                host_num = 1
            else:
                current_voice_name = TTS_VOICE_NAME_ALT
                host_num = 2

            logger.debug(f"  Generating line {count+1} (Host {host_num}, Voice: {current_voice_name}): '{line[:50]}...'")

            synthesis_input = texttospeech.SynthesisInput(text=line)
            voice = texttospeech.VoiceSelectionParams(language_code="en-US", name=current_voice_name)
            audio_config = texttospeech.AudioConfig(audio_encoding=AUDIO_ENCODING)

            try:
                # *** Request Timestamps ***
                response = tts_client.synthesize_speech(
                    input=synthesis_input,
                    voice=voice,
                    audio_config=audio_config,
                    # Add this parameter to get word-level timing
                    enable_time_pointing=[texttospeech.SynthesizeSpeechRequest.TimepointType.WORD]
                )

                # Save audio segment temporarily
                temp_filename = os.path.join(temp_dir, f"part-{count}.mp3")
                with open(temp_filename, "wb") as out:
                    out.write(response.audio_content)

                # Load segment with pydub
                segment = AudioSegment.from_mp3(temp_filename)
                combined_audio_segments.append(segment)

                # *** Process and Store Timestamps ***
                if response.timepoints:
                    logger.debug(f"    -> Received {len(response.timepoints)} timepoints for line {count+1}.")
                    for point in response.timepoints:
                        all_word_timestamps.append({
                            "word": point.word,
                            "start_time_s": current_time_offset_s + point.time_seconds
                        })
                    # Update time offset for the next segment
                    current_time_offset_s += segment.duration_seconds + (AUDIO_SILENCE_MS / 1000.0)
                else:
                    logger.warning(f"    -> No timepoints received for line {count+1}. Timing accuracy might be affected.")
                    # Still update offset based on duration if no timepoints
                    current_time_offset_s += segment.duration_seconds + (AUDIO_SILENCE_MS / 1000.0)

                os.remove(temp_filename) # Clean up temp file immediately

            except google_exceptions.GoogleAPIError as api_error:
                logger.error(f"    TTS API Error for line {count+1}: {api_error}. Skipping segment.")
            except Exception as line_err:
                logger.error(f"    Error generating/processing TTS for line {count+1}: {line_err}. Skipping segment.")

        # --- Combine Audio Segments with Pydub ---
        if not combined_audio_segments:
            logger.error("No audio segments were successfully generated.")
            return {"error": "Failed to generate any audio segments."}

        logger.info(f"Combining {len(combined_audio_segments)} audio segments...")
        # Start with silence, then add segments and silence between them
        full_audio = AudioSegment.silent(duration=AUDIO_SILENCE_MS)
        for segment in combined_audio_segments:
            full_audio += segment + AudioSegment.silent(duration=AUDIO_SILENCE_MS)

        # --- Export Combined Audio ---
        logger.info("Exporting combined audio...")
        buffer = io.BytesIO()
        full_audio.export(buffer, format="mp3")
        audio_bytes = buffer.getvalue()
        logger.info(f"Combined audio generated ({len(audio_bytes)} bytes).")

        # --- Save Final Audio to GCS ---
        timestamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
        blob_name = f"generated/audio/game_{game_pk}/dialogue_audio_{timestamp}.mp3"
        audio_uri = save_audio_to_gcs(audio_bytes, GCS_BUCKET_GENERATED_AUDIO, blob_name)

    except Exception as e:
        logger.error(f"Error during multi-speaker audio generation/combination: {e}", exc_info=True)
        return {"error": f"Failed during multi-speaker audio process: {e}"}
    finally:
        # --- Cleanup Temporary Directory ---
        try:
            if os.path.exists(temp_dir):
                 shutil.rmtree(temp_dir) # Use shutil to remove non-empty dir
                 logger.info(f"Removed temporary directory: {temp_dir}")
        except OSError as e:
             logger.warning(f"Could not remove temporary directory {temp_dir}: {e}")


    node_duration = time.time() - node_start_time
    logger.info(f"--- Multi-Speaker Audio Node Summary ---")
    logger.info(f"  Duration: {node_duration:.2f}s")
    logger.info(f"  Dialogue Lines Processed: {len(dialogue_lines)}")
    logger.info(f"  Segments Combined: {len(combined_audio_segments)}")
    logger.info(f"  Word Timestamps Captured: {len(all_word_timestamps)}")
    logger.info(f"  Generated Audio URI: {audio_uri}")

    # *** Return timestamps along with URI ***
    return {
        "generated_audio_uri": audio_uri,
        "word_timestamps": all_word_timestamps # Store the collected timestamps
    }

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

# --- Define the Aggregation Node ---
def aggregate_final_output_node(state: AgentState) -> Dict[str, Any]:
    """Combines final text, static assets, generated images, and generated videos."""
    logger.info("--- Aggregate Final Output Node ---")
    final_draft = state.get("draft")
    retrieved_static_data = state.get("retrieved_image_data") or []
    generated_image_data = state.get("generated_visual_assets") or []
    generated_video_data = state.get("generated_video_assets") or [] # Get video data

    # *** ADD DEBUG LOGGING HERE ***
    logger.debug(f"Aggregate Node Received State Keys: {list(state.keys())}")
    logger.debug(f"Aggregate Node - Type of generated_video_assets: {type(generated_video_data)}")
    logger.debug(f"Aggregate Node - Length of generated_video_assets: {len(generated_video_data)}")
    if generated_video_data:
        logger.debug(f"Aggregate Node - First video asset received: {generated_video_data[0]}")
    # *** END DEBUG LOGGING ***

    all_image_assets = retrieved_static_data + generated_image_data

    logger.info(f"Aggregating final output: Text script, "
                f"{len(retrieved_static_data)} static assets, "
                f"{len(generated_image_data)} generated images, "
                f"{len(generated_video_data)} generated videos.")

    output = {
        "generated_content": final_draft,
        "all_image_assets": all_image_assets, # Combined static and generated images
        "all_video_assets": generated_video_data # List of generated videos
    }

    if state.get("error"):
        output["error"] = state.get("error")

    return output

# --- Near the top with other prompts ---

# Prompt for the *first* draft generation
GENERATOR_PROMPT_DIALOGUE_TEMPLATE = """
You are an expert MLB analyst and podcast script writer. Your task is to create an engaging **two-host dialogue script** discussing an MLB game based on the user's request and plan.

User Request: "{task}"
Plan:
{plan}

Available Data:
Structured:
```json
{structured_data_json}
Narrative Context (Summaries, Snippets, Research):
{narrative_context_str}

Instructions:

- Dialogue Format: Write a conversation between two hosts (imagine Host 1 and Host 2).

- Strict Alternation: Each line of the script MUST represent one host speaking, alternating strictly between Host 1 and Host 2.

- NO Speaker Labels: CRITICAL: Do NOT include speaker labels like "Host 1:", "Host 2:", or any character names. Just write the raw dialogue line for each speaker's turn.

- Engaging Conversation: Make the dialogue sound natural, with back-and-forth reactions, questions, and analysis. Avoid just reading stats.

- Synthesize Data: Weave the structured data and narrative context naturally into the conversation.

- Storytelling: Build a narrative, highlight key moments, explain significance, and ensure the dialogue flows logically.

- Fulfill Request: Ensure the dialogue fully addresses the original user request: "{task}".

Output ONLY the raw dialogue script, with each speaker's line on a new line.
"""

GENERATOR_PROMPT_REFINED_DIALOGUE_TEMPLATE = """
You are an expert MLB analyst and podcast script writer revising a two-host dialogue script.
Original user request: "{task}"
Original Plan:
{plan}

You previously generated a draft, and received the following critique:
Critique:
{critique}

Based on the critique AND the available data (including internal data and external web search results), revise the dialogue script. Utilize all information below:

Structured Data:

{structured_data_json}
Narrative Context (Summaries, Play Snippets, Internal & Web Research based on Critique):
{narrative_context_str}

Instructions:

- Address the Critique: Explicitly incorporate the feedback, improving the dialogue flow, accuracy, analysis, or engagement as suggested. Use web search results where relevant.

- Maintain Dialogue Format: Ensure the revised script remains a conversation between two hosts.

- Strict Alternation: Each line MUST represent one host speaking, alternating strictly.

- NO Speaker Labels: CRITICAL: Do NOT include speaker labels like "Host 1:", "Host 2:".

- Synthesize ALL Data: Combine structured facts/stats with narrative context naturally within the dialogue.

- Deep Storytelling: Connect stats to game flow, explain significance, highlight key moments/matchups.

- Fulfill Original Task: Ensure the final output clearly answers the user's request: "{task}".

Output ONLY the revised raw dialogue script, with each speaker's line on a new line.
"""

def generate_node_refined(state: AgentState) -> Dict[str, Any]:
    """Generates or revises content as a dialogue script based on data, plan, and critique."""
    logger.info(f"--- Content Generation/Revision Node (Dialogue - Revision: {state.get('revision_number', 0)}) ---")

    # Assume variables like task, plan, critique, structured_data, narrative_context are extracted from state here
    task = state.get('task')
    plan = state.get('plan')
    critique = state.get('critique') # Might be None on first pass
    structured_data = state.get('structured_data')
    narrative_context = state.get('narrative_context', [])
    # ...(rest of the initial variable assignments: task, plan, critique, etc.)...

    if state.get("error"):
        return {"error": state.get("error")}
    if not plan:
        return {"error": "Plan missing."}

    structured_data_json = json.dumps(structured_data, indent=2, default=str) if structured_data else "{}"
    narrative_context_str = "\n---\n".join(narrative_context) if narrative_context else "No narrative context available."

    # Use different prompts for first draft vs revision
    if critique and "excellent" not in critique.lower(): # Revision prompt
        prompt = GENERATOR_PROMPT_REFINED_DIALOGUE_TEMPLATE.format( # Use REFINED DIALOGUE prompt
            task=task,
            plan=plan,
            critique=critique,
            structured_data_json=structured_data_json,
            narrative_context_str=narrative_context_str
        )
    else: # First draft prompt (or if critique was positive)
        prompt = GENERATOR_PROMPT_DIALOGUE_TEMPLATE.format( # Use INITIAL DIALOGUE prompt
            task=task,
            plan=plan,
            structured_data_json=structured_data_json,
            narrative_context_str=narrative_context_str
        )

    try:
        response = model.invoke(prompt)
        new_draft = response.content.strip() # Strip leading/trailing whitespace

        # --- Add basic validation for dialogue format ---
        lines = [line for line in new_draft.splitlines() if line.strip()]
        if not lines:
            logger.error("Generator returned an empty script.")
            return {"error": "Generated script was empty.", "draft": ""}
        # Quick check for common speaker label patterns (e.g., "Host 1:", "Name:")
        if any(":" in line.split(" ")[0] and len(line.split(" ")[0]) < 15 for line in lines[:5]): # Check first word of first 5 lines
             logger.warning("Generated script might contain speaker labels despite instructions. Attempting to proceed anyway.")
             # Potentially add logic here to strip labels if they consistently appear

        logger.info(f"Generated/Revised Dialogue Draft (first 100 chars): {new_draft[:100]}...")
        current_revision = state.get('revision_number', 0)
        return {
            "draft": new_draft,
            "revision_number": current_revision + 1
            }
    except Exception as e:
        logger.error(f"Error in generate_node_refined (dialogue): {e}", exc_info=True)
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
workflow.add_node("analyze_script_for_visual_prompts", analyze_script_for_visual_prompts_node)
workflow.add_node("generate_visuals", generate_visuals_node)
workflow.add_node("critique_visuals", critique_visuals_node)
workflow.add_node("generate_video_clips", generate_video_clips_node)
workflow.add_node("aggregate_final_output", aggregate_final_output_node)
workflow.add_node("generate_audio", generate_audio_node)

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
workflow.add_edge("retrieve_images", "analyze_script_for_visual_prompts")

# --- Add Edges for the NEW Visual Generation Loop ---
workflow.add_edge("analyze_script_for_visual_prompts", "generate_visuals")
workflow.add_edge("generate_visuals", "critique_visuals") # Always critique after generating

# --- Add edge from video generation to final aggregation ---
workflow.add_edge("generate_video_clips", "generate_audio")


# Add the conditional edge for the visual loop
workflow.add_conditional_edges(
    "critique_visuals",       # Node providing the condition basis
    should_continue_visuals,  # Function to decide path
    {
        "generate_visuals": "generate_visuals",         # Loop back to generate more
        # NEW Path: If images are sufficient, generate VIDEOS
        "aggregate_final_output": "generate_video_clips"
    }
)


# NEW: Edge from the asset aggregation node to the NEW audio node
workflow.add_edge("generate_audio","aggregate_final_output")

# NEW: Final edge from audio generation to END
workflow.add_edge("aggregate_final_output", END)
memory = MemorySaver() # Optional: Add if chat history/memory is needed
app = workflow.compile(checkpointer=memory)
app = workflow.compile()

logger.info("Graph compiled successfully.") # Add a log message

# --- START: Graph Visualization Code ---
try:
    logger.info("Attempting to generate graph visualization...")
    # Get the graph structure
    graph_structure = app.get_graph()

    # Generate the PNG image data using Mermaid syntax
    # Ensure you have the necessary optional dependencies installed for LangGraph visualization
    # pip install pygraphviz or other Mermaid rendering tools might be needed if not already present
    png_bytes = graph_structure.draw_mermaid_png()

    # Define the output filename
    output_filename = "mlb_agent_graph_visualization.png"

    # Save the PNG data to a file
    with open(output_filename, "wb") as f:
        f.write(png_bytes)
    logger.info(f"Successfully saved graph visualization to {output_filename}")

except ImportError as ie:
     logger.error(f"Visualization failed: Missing required libraries. Please install necessary dependencies for LangGraph visualization (e.g., 'pip install pygraphviz'). Error: {ie}")
except Exception as e:
    logger.error(f"Failed to generate or save graph visualization: {e}", exc_info=True)
# --- END: Graph Visualization Code ---

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


def load_player_metadata() -> Dict[int, str]:
    """Loads player ID to player name mapping from BigQuery."""
    logger.info("Loading player metadata into memory for lookups...")
    lookup_dict = {} # Initialize as empty dict for this function scope
    try:
        # Ensure BQ client is available (should be initialized globally)
        if 'bq_client' not in globals() or bq_client is None:
             logger.error("BigQuery client not initialized. Cannot load player metadata.")
             return {} # Return empty on critical error

        player_lookup_query = f"SELECT player_id, player_name FROM `{GCP_PROJECT_ID}.{BQ_DATASET_ID}.{PLAYER_METADATA_TABLE_ID}`"
        player_results_df = execute_bq_query(player_lookup_query) # Assumes execute_bq_query is defined above

        if player_results_df is not None and not player_results_df.empty:
            # Use .iterrows() to build the dictionary
            lookup_dict = {int(row['player_id']): row['player_name']
                           for index, row in player_results_df.iterrows()
                           if pd.notna(row['player_id']) and pd.notna(row['player_name'])} # Added check for name too
            logger.info(f"Loaded {len(lookup_dict)} player names into lookup dictionary.")
        else:
            logger.warning("Player metadata query failed or returned no results. Lookup dictionary will be empty.")
    except Exception as meta_err:
         logger.error(f"Failed to load player metadata: {meta_err}. Proceeding with empty lookup.", exc_info=True)
         lookup_dict = {} # Ensure it's an empty dict on error
    return lookup_dict


 # generate_node_refined
# --- Updated Example Usage (at the end of mlb_agent_graph_refined.py) ---
if __name__ == "__main__":
    logger.info("\n--- Running Refined Agent Graph ---")

    # --- Dynamic Game PK ---
    # Choose a default team ID to find the latest game for (e.g., Rangers = 140)
    default_team_id_for_latest = 109
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
        "all_image_assets": [],
        "all_video_assets": [],
        # --- Visual Generation Fields ---
        "visual_generation_prompts": [],
        "generated_visual_assets": [], # Initialize as empty list
        "visual_critique": None,
        "visual_revision_number": 0,
        "max_visual_revisions": 2, # Set the visual loop limit (user requested 2)
        "generated_video_assets": [],
        "generated_audio_uri": [],
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
        elif final_state.get("all_image_assets")  or final_state.get("all_video_assets"): # Check 'draft' as it holds the last generated content

            print("\n--- Final Generated Content & Assets ---")
            print("\n** Script: **")
            print(final_state.get("generated_content", "N/A"))
            print("\n** Visual Assets (Static & Generated): **")
            if final_state.get("all_image_assets"):
               print(json.dumps(final_state["all_image_assets"], indent=2, default=str))
            else:
               print("No visual assets found.")

            print("\n** Video Assets (Generated): **") # Print video assets
            if final_state.get("all_video_assets"):
                print(json.dumps(final_state["all_video_assets"], indent=2, default=str))
            else:
                print("No video assets found.")

            print("\n** Audio Asset (Generated): **") # Print audio asset URI
            if final_state.get("generated_audio_uri"):
                print(final_state.get("generated_audio_uri", "No audio generated or URI missing."))
        else:
             print("\n--- Execution Finished, but no final content/assets found. Check logs. ---")
             # Print selective final state for debugging
             print("Final state snapshot (excluding large data):", {k: v for k, v in final_state.items() if k not in ['structured_data', 'narrative_context', 'draft', 'generated_content', 'all_visual_assets', 'all_video_assets', 'generated_audio_uri']})

    except Exception as e:
         logger.error(f"Error invoking graph: {e}", exc_info=True)
         print(f"\n--- Graph Invocation Error ---")
         print(f"An exception occurred: {e}")