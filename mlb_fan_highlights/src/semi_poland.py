# enhanced_data_ingestion.py
import pandas as pd
import requests
import json
from datetime import datetime, UTC, timedelta
import logging
import time
from typing import Dict, List, Tuple, Any, Optional
import os
import tempfile
import re # For basic keyword matching in key play identification

from ratelimit import limits, sleep_and_retry
from google.cloud import bigquery
from google.api_core.exceptions import BadRequest, NotFound
import vertexai
from vertexai.generative_models import GenerativeModel, Part, GenerationConfig
from vertexai.language_models import TextEmbeddingModel, TextEmbeddingInput

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration (Ensure these match your environment) ---
GCP_PROJECT_ID = "silver-455021"  # <-- REPLACE THIS
GCP_LOCATION = "us-central1"
BQ_LOCATION = "US"
BQ_DATASET_ID = "mlb_rag_data_2024"
BQ_RAG_TABLE_ID = "rag_documents"      # For summaries & play snippets + embeddings
BQ_PLAYS_TABLE_ID = "plays"            # NEW: For structured play-by-play data
BQ_INDEX_NAME = "rag_docs_embedding_idx" # Index on rag_documents table

VERTEX_LLM_MODEL = "gemini-2.0-flash" # Use flash for potentially faster processing
VERTEX_EMB_MODEL = "text-embedding-004"
EMBEDDING_TASK_TYPE = "RETRIEVAL_DOCUMENT"
EMBEDDING_DIMENSIONALITY = 768

MLB_API_CALLS = 9
MLB_API_RATE_LIMIT = 60
VERTEX_LLM_RPM = 180 # Adjust based on model and quotas
VERTEX_EMB_RPM = 1400 # Adjust

NUM_GAMES_PER_TEAM = 1 # Keep low for testing, increase later
MAX_PLAY_SNIPPETS_PER_GAME = 10 # Limit LLM calls for play snippets

# Teams (Shortened)
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


# --- Initialize Clients ---
try:
    bq_client = bigquery.Client(project=GCP_PROJECT_ID)
    vertexai.init(project=GCP_PROJECT_ID, location=GCP_LOCATION)
    llm_model = GenerativeModel(VERTEX_LLM_MODEL)
    emb_model = TextEmbeddingModel.from_pretrained(VERTEX_EMB_MODEL)
    logger.info(f"Initialized Google Cloud clients for project {GCP_PROJECT_ID}")
except Exception as e:
    logger.critical(f"Failed to initialize Google Cloud clients: {e}", exc_info=True)
    exit()

# --- BigQuery Schemas ---
RAG_SCHEMA = [
    bigquery.SchemaField("doc_id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("game_id", "INTEGER", mode="REQUIRED"),
    bigquery.SchemaField("doc_type", "STRING", mode="NULLABLE", description="'game_summary' or 'play_snippet'"),
    bigquery.SchemaField("play_index", "INTEGER", mode="NULLABLE", description="Index of the play within the game (for play_snippet)"),
    bigquery.SchemaField("content", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("embedding", "FLOAT64", mode="REPEATED", description=f"Vector embedding ({EMBEDDING_DIMENSIONALITY} dimensions)"),
    bigquery.SchemaField("metadata", "JSON", mode="NULLABLE"),
    bigquery.SchemaField("last_updated", "TIMESTAMP", mode="NULLABLE"),
]

PLAYS_SCHEMA = [
    bigquery.SchemaField("game_pk", "INTEGER", mode="REQUIRED"),
    bigquery.SchemaField("play_index", "INTEGER", mode="REQUIRED"), # Index within allPlays array
    bigquery.SchemaField("at_bat_index", "INTEGER", mode="NULLABLE"),
    bigquery.SchemaField("inning", "INTEGER", mode="NULLABLE"),
    bigquery.SchemaField("is_top_inning", "BOOLEAN", mode="NULLABLE"),
    bigquery.SchemaField("half_inning", "STRING", mode="NULLABLE"), # e.g., "top", "bottom"
    bigquery.SchemaField("event_type", "STRING", mode="NULLABLE"), # e.g., "strikeout", "home_run"
    bigquery.SchemaField("description", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("rbi", "INTEGER", mode="NULLABLE"),
    bigquery.SchemaField("away_score", "INTEGER", mode="NULLABLE"),
    bigquery.SchemaField("home_score", "INTEGER", mode="NULLABLE"),
    bigquery.SchemaField("is_scoring_play", "BOOLEAN", mode="NULLABLE"),
    bigquery.SchemaField("has_out", "BOOLEAN", mode="NULLABLE"),
    bigquery.SchemaField("outs_before_play", "INTEGER", mode="NULLABLE"),
    bigquery.SchemaField("balls_before_play", "INTEGER", mode="NULLABLE"),
    bigquery.SchemaField("strikes_before_play", "INTEGER", mode="NULLABLE"),
    bigquery.SchemaField("batter_id", "INTEGER", mode="NULLABLE"),
    bigquery.SchemaField("pitcher_id", "INTEGER", mode="NULLABLE"),
    bigquery.SchemaField("runners_before", "JSON", mode="NULLABLE"), # Store runner state before
    bigquery.SchemaField("runners_after", "JSON", mode="NULLABLE"), # Store runner state after
    bigquery.SchemaField("pitch_data", "JSON", mode="NULLABLE"), # Store pitch details if available
    bigquery.SchemaField("hit_data", "JSON", mode="NULLABLE"), # Store hit details if available
    bigquery.SchemaField("play_start_time", "TIMESTAMP", mode="NULLABLE"),
    bigquery.SchemaField("play_end_time", "TIMESTAMP", mode="NULLABLE"),
    # Add more fields as needed based on deep dive analysis
]

# --- Utility Functions (Keep from original, including ensure_dataset_exists) ---
@sleep_and_retry
@limits(calls=MLB_API_CALLS, period=MLB_API_RATE_LIMIT)
def call_mlb_api(url: str) -> Dict:
    # (Same as your provided function)
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

def ensure_dataset_exists(client: bigquery.Client, dataset_id: str):
    # (Same as your provided function)
    full_dataset_id = f"{client.project}.{dataset_id}"
    try:
        client.get_dataset(full_dataset_id)
        logger.info(f"Dataset {full_dataset_id} already exists")
    except NotFound:
        dataset = bigquery.Dataset(full_dataset_id)
        dataset.location = BQ_LOCATION
        dataset = client.create_dataset(dataset, timeout=30)
        logger.info(f"Created dataset {full_dataset_id}")
    except Exception as e:
        logger.error(f"Error checking/creating dataset {full_dataset_id}: {e}")
        raise

def create_bq_table(client: bigquery.Client, dataset_id: str, table_id: str, schema: List[bigquery.SchemaField]):
    """Generic function to create a BQ table if it doesn't exist."""
    full_table_id = f"{client.project}.{dataset_id}.{table_id}"
    try:
        client.get_table(full_table_id)
        logger.info(f"Table {full_table_id} already exists.")
    except NotFound:
        logger.info(f"Table {full_table_id} not found, creating...")
        table = bigquery.Table(full_table_id, schema=schema)
        try:
            client.create_table(table)
            logger.info(f"Created table {full_table_id}")
        except Exception as e:
            logger.error(f"Failed to create table {full_table_id}: {e}")
            raise
    except Exception as e:
         logger.error(f"Error checking table {full_table_id}: {e}")
         raise

# Rate limited LLM and Embedding calls (Keep from original)
@sleep_and_retry
@limits(calls=VERTEX_LLM_RPM, period=60)
def call_vertex_llm(prompt: str) -> Optional[str]:
    # (Same as your provided function)
    try:
        generation_config = GenerationConfig(temperature=0.2, max_output_tokens=1024)
        response = llm_model.generate_content(prompt, generation_config=generation_config)
        if response.candidates and response.candidates[0].content.parts:
            text_parts = [part.text for part in response.candidates[0].content.parts if hasattr(part, 'text')]
            return "\n".join(text_parts).strip() if text_parts else None
        else:
            block_reason = response.prompt_feedback.block_reason if response.prompt_feedback else 'Unknown'
            logger.warning(f"LLM generation blocked/empty. Reason: {block_reason}.")
            return None
    except Exception as e:
        logger.error(f"Error calling Vertex AI LLM: {e}.")
        return None

@sleep_and_retry
@limits(calls=VERTEX_EMB_RPM, period=60)
def call_vertex_embedding(text_inputs: List[str]) -> List[Optional[List[float]]]:
    # (Same as your provided function, ensure it handles batch limits < 250 if needed)
    results = []
    batch_size = 200 # Stay under the limit
    try:
        all_embeddings = []
        for i in range(0, len(text_inputs), batch_size):
            batch = text_inputs[i:i + batch_size]
            instances = [TextEmbeddingInput(text=text, task_type=EMBEDDING_TASK_TYPE) for text in batch]
            kwargs = {"output_dimensionality": EMBEDDING_DIMENSIONALITY} if EMBEDDING_TASK_TYPE != "RETRIEVAL_DOCUMENT" else {}
            embeddings_batch = emb_model.get_embeddings(instances, **kwargs)
            all_embeddings.extend([emb.values for emb in embeddings_batch])
            if len(text_inputs) > batch_size:
                 logger.info(f"Embedded batch {i // batch_size + 1}, pausing briefly...")
                 time.sleep(1) # Small pause between large batches


        # Basic validation
        if not all(len(emb) == EMBEDDING_DIMENSIONALITY for emb in all_embeddings if emb):
             logger.warning(f"Embeddings have unexpected dimensionality. Expected {EMBEDDING_DIMENSIONALITY}.")

        return all_embeddings

    except Exception as e:
        logger.error(f"Error calling Vertex AI Embedding API for batch size {len(text_inputs)}: {e}", exc_info=True)
        return [None] * len(text_inputs)


def upload_to_bigquery(df: pd.DataFrame, table_id: str, schema: List[bigquery.SchemaField]):
    # (Same as your provided function using tempfile)
    if df.empty:
        logger.info(f"DataFrame is empty, skipping upload to {table_id}.")
        return

    full_table_id_path = f"{GCP_PROJECT_ID}.{BQ_DATASET_ID}.{table_id}" # Use full path for BQ Client

    # Clean data for BQ JSON/Timestamp compatibility before writing to temp file
    for col in df.columns:
        if isinstance(df[col].dtype, pd.ArrowDtype) and pd.types.is_list(df[col].dtype.pyarrow_dtype):
            # Handle potential Arrow list types (like embeddings) - Ensure they are basic lists
            df[col] = df[col].apply(lambda x: list(x) if x is not None else None)
        elif col == 'metadata' and not df[col].isnull().all():
             # Ensure metadata is dict for JSON conversion
             df[col] = df[col].apply(lambda x: x if isinstance(x, dict) else (json.loads(x) if isinstance(x, str) else {}))
        elif col.endswith('_time') or col == 'last_updated': # Handle timestamps
            # Convert valid datetimes to BQ-compatible ISO string format
            df[col] = pd.to_datetime(df[col], errors='coerce', utc=True).dt.strftime('%Y-%m-%dT%H:%M:%S.%fZ')


    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.ndjson', encoding='utf-8') as temp_f:
        temp_file_path = temp_f.name
        logger.info(f"Writing DataFrame to temporary NDJSON file: {temp_file_path}")
        df.to_json(
            temp_f,
            orient='records',
            lines=True,
            force_ascii=False,
            date_format='iso' # Ensures timestamps are written correctly
        )
        temp_f.flush()

    logger.info(f"Finished writing. Starting BigQuery load from file to {full_table_id_path}.")
    job_config = bigquery.LoadJobConfig(
        write_disposition="WRITE_APPEND",
        schema=schema,
        source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
    )

    try:
        with open(temp_file_path, "rb") as source_file:
            job = bq_client.load_table_from_file(source_file, full_table_id_path, job_config=job_config)
        job.result()
        table = bq_client.get_table(full_table_id_path)
        logger.info(f"Loaded {job.output_rows} rows to {table_id}. Total rows: {table.num_rows}.")
    except BadRequest as e:
        logger.error(f"BigQuery BadRequest loading to {table_id}: {e}", exc_info=True)
        if hasattr(e, 'errors'): logger.error(f"  Errors: {e.errors}")
    except Exception as e:
        logger.error(f"Generic error loading to {table_id}: {e}", exc_info=True)
    finally:
        try:
            os.remove(temp_file_path)
            logger.info(f"Removed temporary file: {temp_file_path}")
        except OSError as e:
            logger.error(f"Error removing temp file {temp_file_path}: {e}")

def create_bq_vector_index(client: bigquery.Client, dataset_id: str, table_id: str, index_name: str):
    # (Same as your provided function)
    full_table_id = f"{client.project}.{dataset_id}.{table_id}"
    index_check_sql = f"SELECT index_name FROM `{client.project}.{dataset_id}`.INFORMATION_SCHEMA.VECTOR_INDEXES WHERE table_name = '{table_id}' AND index_name = '{index_name}';"
    create_index_sql = f"CREATE OR REPLACE VECTOR INDEX `{index_name}` ON `{full_table_id}`(embedding) OPTIONS(distance_type='COSINE', index_type='IVF');"
    try:
        results = client.query(index_check_sql).result()
        if results.total_rows > 0:
            logger.info(f"Vector index {index_name} already exists on {full_table_id}.")
            return
        logger.info(f"Creating vector index {index_name} on {full_table_id}...")
        client.query(create_index_sql).result()
        logger.info(f"Vector index {index_name} creation initiated.")
    except Exception as e:
        logger.error(f"Error checking or creating vector index {index_name}: {e}")

# --- NEW/Enhanced Processing Functions ---

def get_recent_game_ids(team_id: int, season: int = 2025, num_games: int = 50) -> List[int]:
    # (Keep same as original)
    url = f'https://statsapi.mlb.com/api/v1/schedule?sportId=1&season={season}&teamId={team_id}&fields=dates,games,gamePk,officialDate,status,detailedState'
    schedule_data = call_mlb_api(url)
    game_ids = []
    if schedule_data and 'dates' in schedule_data:
        all_games = []
        for date_entry in schedule_data.get('dates', []):
            for game in date_entry.get('games', []):
                 if game.get('status', {}).get('detailedState') == 'Final':
                    all_games.append({'game_id': game.get('gamePk'), 'date': game.get('officialDate')})
        all_games.sort(key=lambda x: x['date'], reverse=True)
        game_ids = [game['game_id'] for game in all_games[:num_games] if game['game_id']]
    if not game_ids: logger.warning(f"No recent final game IDs for team {team_id}, season {season}.")
    return game_ids

def get_full_game_data(game_pk: int) -> Dict:
    # (Keep same as original)
    url = f'https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live'
    logger.info(f"Fetching full data for game {game_pk}")
    data = call_mlb_api(url)
    if not data: logger.error(f"Failed to fetch/empty data for game {game_pk}")
    return data

def generate_game_summary_and_metadata(game_pk: int, game_data: Dict) -> Optional[Tuple[str, Dict]]:
    # (Keep same as original, ensures basic summary is generated)
     if not game_data or 'gameData' not in game_data or 'liveData' not in game_data: return None
     try:
         game_info = game_data.get('gameData', {}); live_info = game_data.get('liveData', {})
         game_pk = game_info.get('game', {}).get('pk', game_pk)
         status = game_info.get('status', {}).get('detailedState')
         if status != 'Final': return None
         home_team_data = game_info.get('teams', {}).get('home', {})
         away_team_data = game_info.get('teams', {}).get('away', {})
         home_team = home_team_data.get('name', 'N/A'); away_team = away_team_data.get('name', 'N/A')
         date = game_info.get('datetime', {}).get('officialDate')
         venue = game_info.get('venue', {}).get('name', 'N/A')
         linescore = live_info.get('linescore', {})
         home_score = linescore.get('teams', {}).get('home', {}).get('runs', 0)
         away_score = linescore.get('teams', {}).get('away', {}).get('runs', 0)
         innings_data = linescore.get('innings', [])
         scoring_plays = [p for p in live_info.get('plays', {}).get('allPlays', []) if p.get('about', {}).get('isScoringPlay')]
         key_plays_info = [f"Inning {p.get('about',{}).get('inning','?')}({p.get('about',{}).get('halfInning','')}): {p.get('result',{}).get('description','N/A')}" for p in scoring_plays]
         key_plays_str = "\n".join(key_plays_info) if key_plays_info else "No scoring plays listed."

         metadata = { "date": date, "season": int(date[:4]) if date else None,
                      "home_team_id": int(home_team_data.get('id')) if home_team_data.get('id') else None,
                      "home_team_name": home_team, "away_team_id": int(away_team_data.get('id')) if away_team_data.get('id') else None,
                      "away_team_name": away_team, "home_score": int(home_score), "away_score": int(away_score),
                      "venue_name": venue, "status": status, "innings": len(innings_data)}

         prompt = f"Summarize MLB game: {away_team} ({away_score}) vs {home_team} ({home_score}) on {date} at {venue}. Key moments:\n{key_plays_str}\nProvide a 2-4 sentence overview."
         logger.info(f"Generating summary for game {game_pk}")
         summary = call_vertex_llm(prompt)
         return (summary, metadata) if summary else None
     except Exception as e:
         logger.error(f"Error generating summary for game {game_pk}: {e}", exc_info=True)
         return None

def extract_structured_plays(game_pk: int, game_data: Dict) -> pd.DataFrame:
    """Extracts structured play-by-play data into a DataFrame."""
    all_plays_data = []
    plays = game_data.get('liveData', {}).get('plays', {}).get('allPlays', [])

    for play_index, play in enumerate(plays):
        about = play.get('about', {})
        count = play.get('count', {})
        matchup = play.get('matchup', {})
        result = play.get('result', {})
        runners = play.get('runners', []) # All runners involved in the play

        # Extract runner state before (approximated by postOnX from *previous* play if needed, complex logic)
        # For simplicity here, we store the state *after* this play. More sophisticated state tracking is possible.
        runners_after_state = {
            "first": matchup.get('postOnFirst', {}).get('id') if matchup.get('postOnFirst') else None,
            "second": matchup.get('postOnSecond', {}).get('id') if matchup.get('postOnSecond') else None,
            "third": matchup.get('postOnThird', {}).get('id') if matchup.get('postOnThird') else None,
        }

        # Find the primary pitch event (usually the last one if it's a pitch)
        pitch_event = None
        hit_event_data = None
        for ev in play.get('playEvents', []):
             if ev.get('isPitch'):
                 pitch_event = ev # Take the last pitch event as representative
             if ev.get('hitData'):
                  hit_event_data = ev.get('hitData')

        play_dict = {
            "game_pk": game_pk,
            "play_index": play_index,
            "at_bat_index": about.get('atBatIndex'),
            "inning": about.get('inning'),
            "is_top_inning": about.get('isTopInning'),
            "half_inning": about.get('halfInning'),
            "event_type": result.get('eventType'),
            "description": result.get('description'),
            "rbi": result.get('rbi'),
            "away_score": result.get('awayScore'),
            "home_score": result.get('homeScore'),
            "is_scoring_play": about.get('isScoringPlay'),
            "has_out": result.get('isOut'),
            "outs_before_play": count.get('outs'), # Outs *before* the current pitch/action
            "balls_before_play": count.get('balls'),
            "strikes_before_play": count.get('strikes'),
            "batter_id": matchup.get('batter', {}).get('id'),
            "pitcher_id": matchup.get('pitcher', {}).get('id'),
            "runners_before": None, # Placeholder - requires state tracking between plays
            "runners_after": runners_after_state, # State *after* the play completes
            "pitch_data": pitch_event.get('pitchData') if pitch_event else None,
            "hit_data": hit_event_data,
            "play_start_time": pd.to_datetime(about.get('startTime'), errors='coerce', utc=True),
            "play_end_time": pd.to_datetime(about.get('endTime'), errors='coerce', utc=True),
        }
        all_plays_data.append(play_dict)

    if not all_plays_data:
        return pd.DataFrame(columns=[f.name for f in PLAYS_SCHEMA]) # Return empty DF with correct columns

    return pd.DataFrame(all_plays_data)

def identify_key_plays(plays_df: pd.DataFrame, max_snippets: int) -> List[int]:
    """Identifies indices of 'key' plays based on simple criteria."""
    key_indices = set()

    # 1. Scoring Plays
    scoring_plays = plays_df[plays_df['is_scoring_play'] == True]
    key_indices.update(scoring_plays.index.tolist())

    # 2. Home Runs
    home_runs = plays_df[plays_df['event_type'] == 'home_run']
    key_indices.update(home_runs.index.tolist())

    # 3. High-Leverage Situtations (Simple Heuristic: Runners in scoring position, late innings)
    # More complex leverage index calculation is possible but involved
    late_innings = plays_df['inning'] >= 7
    runners_on_2nd = plays_df['runners_after'].apply(lambda x: x is not None and x.get('second') is not None)
    runners_on_3rd = plays_df['runners_after'].apply(lambda x: x is not None and x.get('third') is not None)
    risp = runners_on_2nd | runners_on_3rd
    high_leverage_candidates = plays_df[late_innings & risp]
    key_indices.update(high_leverage_candidates.index.tolist())

    # 4. Strikeouts with RISP or Bases Loaded? (Example)
    bases_loaded = plays_df['runners_after'].apply(lambda x: x and x.get('first') and x.get('second') and x.get('third'))
    strikeouts_high_leverage = plays_df[(plays_df['event_type'] == 'strikeout') & (risp | bases_loaded)]
    key_indices.update(strikeouts_high_leverage.index.tolist())

    # Limit and return sorted indices
    sorted_indices = sorted(list(key_indices))
    return sorted_indices[:max_snippets]

def generate_play_snippet_and_metadata(game_pk: int, play_data: pd.Series, game_metadata: Dict) -> Optional[Tuple[str, Dict]]:
    """Generates a narrative snippet for a single key play using an LLM."""
    try:
        play_desc = play_data.get('description', 'N/A')
        inning = play_data.get('inning')
        half = play_data.get('half_inning', '')
        outs = play_data.get('outs_before_play', 0)
        away_score = play_data.get('away_score', 0)
        home_score = play_data.get('home_score', 0)
        batter_id = play_data.get('batter_id') # Need to lookup name if desired
        pitcher_id = play_data.get('pitcher_id') # Need to lookup name if desired

        # Basic context
        context = f"""
        In Game {game_pk} ({game_metadata.get('away_team_name')} vs {game_metadata.get('home_team_name')} on {game_metadata.get('date')}),
        during the {half} of inning {inning} with {outs} outs, the score was Away {away_score}, Home {home_score}.
        Play Description from API: {play_desc}
        """

        prompt = f"""
        Analyze the following MLB play context and generate a concise narrative snippet (1-2 sentences) describing the play and its significance. Focus on clear action and impact.

        Context:
        {context}

        Snippet:
        """
        logger.info(f"Generating snippet for game {game_pk}, play index {play_data.name}")
        snippet = call_vertex_llm(prompt)

        if snippet:
            play_metadata = {
                "game_date": game_metadata.get('date'),
                "inning": int(inning) if inning else None,
                "half_inning": half,
                "batter_id": int(batter_id) if batter_id else None,
                "pitcher_id": int(pitcher_id) if pitcher_id else None,
                "event_type": play_data.get('event_type'),
                "is_scoring": play_data.get('is_scoring_play'),
                # Add more relevant play metadata if needed
            }
            # Merge with game metadata for richer context? Optional.
            # play_metadata.update(game_metadata) # Be mindful of size/redundancy
            return snippet, play_metadata
        else:
            logger.warning(f"LLM failed to generate snippet for game {game_pk}, play {play_data.name}")
            return None

    except Exception as e:
        logger.error(f"Error generating play snippet for game {game_pk}, play {play_data.name}: {e}", exc_info=True)
        return None


# --- In enhanced_data_ingestion.py ---

# (Keep all other functions and imports the same)

def process_game_for_rag_enhanced(game_pk: int, rag_table_id: str, plays_table_id: str):
    """Fetches, processes, summarizes, embeds, and uploads data for a single game to RAG and Plays tables."""
    game_data = get_full_game_data(game_pk)
    if not game_data:
        logger.warning(f"Skipping game {game_pk} due to fetch failure.")
        return

    # 1. Process Game Summary
    summary_result = generate_game_summary_and_metadata(game_pk, game_data)
    game_summary_metadata = {} # Default empty dict
    summary_text = None
    if summary_result:
        summary_text, game_summary_metadata = summary_result
    else:
        logger.warning(f"Game {game_pk} summary generation failed, but proceeding with play data.")
        # game_summary_metadata remains {}

    # 2. Process Structured Plays (Extract)
    plays_df = extract_structured_plays(game_pk, game_data)
    key_play_indices = []

    if not plays_df.empty:
        logger.info(f"Extracted {len(plays_df)} structured plays for game {game_pk}.")
        # 3. Identify Key Plays (BEFORE JSON conversion)
        try:
            # Ensure runners_after exists and handle potential non-dict entries gracefully
            if 'runners_after' in plays_df.columns:
                 plays_df['runners_after'] = plays_df['runners_after'].apply(lambda x: x if isinstance(x, dict) else {})
            else:
                 # If the column doesn't exist from extraction, add an empty dict column
                 # This prevents errors in identify_key_plays if it expects the column
                 plays_df['runners_after'] = [{}] * len(plays_df)

            key_play_indices = identify_key_plays(plays_df, MAX_PLAY_SNIPPETS_PER_GAME)
            logger.info(f"Identified {len(key_play_indices)} key plays for snippet generation in game {game_pk}.")
        except Exception as e_ident:
             logger.error(f"Error during identify_key_plays for game {game_pk}: {e_ident}", exc_info=True)
             key_play_indices = [] # Proceed without snippets if identification fails


        # Create a copy for upload
        plays_df_for_upload = plays_df.copy()
        # Clean JSON columns FOR BQ UPLOAD
        for col in ['runners_before', 'runners_after', 'pitch_data', 'hit_data']:
            if col in plays_df_for_upload.columns:
                 plays_df_for_upload[col] = plays_df_for_upload[col].apply(lambda x: json.dumps(x) if isinstance(x, (dict, list)) else None)
        upload_to_bigquery(plays_df_for_upload, plays_table_id, PLAYS_SCHEMA)
    else:
        logger.warning(f"No plays extracted for game {game_pk}.")

    # 4. Process Key Play Snippets & Summary
    texts_to_embed = []
    doc_metadata_list = []

    # Add game summary
    if summary_text and game_summary_metadata:
         doc_id_summary = f"game_{game_pk}_summary"
         texts_to_embed.append(summary_text)
         doc_metadata_list.append({
             "doc_id": doc_id_summary, "game_id": game_pk, "doc_type": "game_summary",
             "play_index": None, # Explicitly None for summary
             "content": summary_text, "metadata": game_summary_metadata,
             "last_updated": datetime.now(UTC)
         })

    # Generate snippets
    if not plays_df.empty: # Re-check if plays_df was created
        for play_idx in key_play_indices:
            if play_idx < len(plays_df):
                play_data_series = plays_df.iloc[play_idx]
                snippet_result = generate_play_snippet_and_metadata(game_pk, play_data_series, game_summary_metadata)
                if snippet_result:
                    snippet_text, snippet_metadata = snippet_result
                    doc_id_snippet = f"game_{game_pk}_play_{play_idx}"
                    texts_to_embed.append(snippet_text)
                    doc_metadata_list.append({
                        "doc_id": doc_id_snippet, "game_id": game_pk, "doc_type": "play_snippet",
                        "play_index": int(play_idx), # Cast to standard Python int here
                        "content": snippet_text, "metadata": snippet_metadata,
                        "last_updated": datetime.now(UTC)
                    })
                else:
                    logger.warning(f"Skipping snippet generation for play index {play_idx} in game {game_pk}.")
            else:
                logger.warning(f"Key play index {play_idx} out of bounds for plays_df (len {len(plays_df)}) in game {game_pk}.")

    else:
        logger.info(f"Skipping snippet generation as plays_df is empty for game {game_pk}.")


    # 5. Batch Embeddings & Upload to RAG table
    if texts_to_embed:
        logger.info(f"Generating embeddings for {len(texts_to_embed)} documents for game {game_pk}...")
        embeddings = call_vertex_embedding(texts_to_embed)

        if embeddings and len(embeddings) == len(doc_metadata_list):
            rag_docs_to_upload = []
            for i, embedding in enumerate(embeddings):
                if embedding:
                    doc_data = doc_metadata_list[i]
                    doc_data["embedding"] = embedding
                    rag_docs_to_upload.append(doc_data)
                else:
                    logger.error(f"Failed to get embedding for doc: {doc_metadata_list[i]['doc_id']}. Skipping.")

            if rag_docs_to_upload:
                rag_df_final = pd.DataFrame(rag_docs_to_upload)

                # ***** FIX: Explicitly set nullable integer type *****
                if 'play_index' in rag_df_final.columns:
                    # Convert to float first to handle potential non-integer strings/None, then to nullable Int64
                    rag_df_final['play_index'] = pd.to_numeric(rag_df_final['play_index'], errors='coerce')
                    rag_df_final['play_index'] = rag_df_final['play_index'].astype('Int64')
                    logger.info("Converted 'play_index' column to nullable Int64 dtype.")
                # *****************************************************

                logger.info(f"Attempting to upload {len(rag_df_final)} RAG documents for game {game_pk} to {rag_table_id}")
                upload_to_bigquery(rag_df_final, rag_table_id, RAG_SCHEMA)
        else:
            logger.error(f"Mismatch in embedding results/metadata list for game {game_pk}. Skipping RAG upload.")
    else:
        logger.info(f"No texts generated for embedding for game {game_pk}.")




def main_enhanced_ingestion():
    """Main function to run the enhanced data ingestion pipeline."""
    start_time = time.time()
    logger.info("Starting ENHANCED MLB RAG data pipeline...")

    dataset_id = BQ_DATASET_ID
    rag_table_id = BQ_RAG_TABLE_ID
    plays_table_id = BQ_PLAYS_TABLE_ID
    full_rag_table_id = f"{GCP_PROJECT_ID}.{dataset_id}.{rag_table_id}"
    full_plays_table_id = f"{GCP_PROJECT_ID}.{dataset_id}.{plays_table_id}"

    # Ensure Dataset and Tables exist
    try:
        ensure_dataset_exists(bq_client, dataset_id)
        create_bq_table(bq_client, dataset_id, rag_table_id, RAG_SCHEMA)
        create_bq_table(bq_client, dataset_id, plays_table_id, PLAYS_SCHEMA)
    except Exception as e:
        logger.critical(f"Failed to ensure BQ dataset/tables exist. Exiting. Error: {e}", exc_info=True)
        return

    all_game_pks_to_process = set()
    logger.info("Gathering recent game IDs...")
    for team_name, team_id in TEAMS.items():
        logger.info(f"Fetching game IDs for {team_name}...")
        try:
            team_game_ids = get_recent_game_ids(team_id, num_games=NUM_GAMES_PER_TEAM)
            all_game_pks_to_process.update(team_game_ids)
            logger.info(f"Found {len(team_game_ids)} recent game(s) for {team_name}.")
            time.sleep(1)
        except Exception as e:
            logger.error(f"Error fetching game IDs for {team_name} (ID: {team_id}): {e}")

    logger.info(f"Total unique recent games to process: {len(all_game_pks_to_process)}")

    processed_count = 0
    for game_pk in all_game_pks_to_process:
        try:
            logger.info(f"--- Processing Game PK: {game_pk} ---")
            process_game_for_rag_enhanced(game_pk, rag_table_id, plays_table_id)
            processed_count += 1
            time.sleep(0.5)
        except Exception as e:
            logger.error(f"Critical error processing game {game_pk}: {e}", exc_info=True)

    # Create Vector Index (after all RAG data is loaded)
    logger.info("Data loading complete. Ensuring vector index exists on RAG table...")
    create_bq_vector_index(bq_client, dataset_id, rag_table_id, BQ_INDEX_NAME)

    end_time = time.time()
    logger.info(f"Enhanced MLB RAG data pipeline finished. Processed {processed_count} games.")
    logger.info(f"Total execution time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main_enhanced_ingestion()