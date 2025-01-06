import os
from google.cloud import bigquery
import requests
import pandas as pd
from ratelimit import limits, sleep_and_retry
import logging
from datetime import datetime, UTC
from typing import Dict, List, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# BigQuery client setup
client = bigquery.Client()

# Rate limiting decorators
CALLS = 100
RATE_LIMIT = 60

@sleep_and_retry
@limits(calls=CALLS, period=RATE_LIMIT)
def call_mlb_api(url: str) -> Dict:
    """Make a rate-limited call to the MLB API"""
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

# Schema definitions
PLAYER_PROFILE_SCHEMA = [
    bigquery.SchemaField("player_id", "INTEGER", mode="REQUIRED"),
    bigquery.SchemaField("team_id", "INTEGER", mode="REQUIRED"),
    bigquery.SchemaField("team_name", "STRING"),
    bigquery.SchemaField("full_name", "STRING"),
    bigquery.SchemaField("first_name", "STRING"),
    bigquery.SchemaField("last_name", "STRING"),
    bigquery.SchemaField("primary_position", "STRING"),
    bigquery.SchemaField("jersey_number", "STRING"),
    bigquery.SchemaField("birth_date", "DATE"),
    bigquery.SchemaField("age", "INTEGER"),
    bigquery.SchemaField("height", "STRING"),
    bigquery.SchemaField("weight", "INTEGER"),
    bigquery.SchemaField("birth_city", "STRING"),
    bigquery.SchemaField("birth_country", "STRING"),
    bigquery.SchemaField("status", "STRING"),
    bigquery.SchemaField("active", "BOOLEAN"),
    bigquery.SchemaField("last_updated", "TIMESTAMP")
]

GAME_DATA_SCHEMA = [
    bigquery.SchemaField("game_id", "INTEGER", mode="REQUIRED"),
    bigquery.SchemaField("game_date", "DATE", mode="REQUIRED"),
    bigquery.SchemaField("season", "INTEGER"),
    bigquery.SchemaField("game_type", "STRING"),
    bigquery.SchemaField("status", "STRING"),
    bigquery.SchemaField("home_team_id", "INTEGER"),
    bigquery.SchemaField("away_team_id", "INTEGER"),
    bigquery.SchemaField("home_team_name", "STRING"),
    bigquery.SchemaField("away_team_name", "STRING"),
    bigquery.SchemaField("home_score", "INTEGER"),
    bigquery.SchemaField("away_score", "INTEGER"),
    bigquery.SchemaField("venue_name", "STRING"),
    bigquery.SchemaField("venue_id", "INTEGER"),
    bigquery.SchemaField("last_updated", "TIMESTAMP")
]

def get_team_roster(team_id: int, season: int = 2024) -> pd.DataFrame:
    """
    Get roster for a specific team and season
    """
    url = f'https://statsapi.mlb.com/api/v1/teams/{team_id}/roster?season={season}'
    roster_data = call_mlb_api(url)
    
    if 'roster' not in roster_data:
        logger.warning(f"No roster data found for team {team_id}")
        return pd.DataFrame()
    
    # Extract player IDs and basic roster info
    roster_list = []
    for player in roster_data['roster']:
        player_info = {
            'player_id': player['person']['id'],
            'position': player['position']['name'],
            'status': player.get('status', {}).get('description', ''),
            'jerseyNumber': player.get('jerseyNumber', '')
        }
        roster_list.append(player_info)
    
    roster_df = pd.DataFrame(roster_list)
    
    # Get detailed player info for each roster member
    player_details = []
    for player_id in roster_df['player_id']:
        player_url = f"https://statsapi.mlb.com/api/v1/people/{player_id}"
        player_data = call_mlb_api(player_url)
        if 'people' in player_data and len(player_data['people']) > 0:
            player_details.append(player_data['people'][0])
    
    # Create detailed player DataFrame
    player_df = pd.json_normalize(player_details)
    
    # Merge roster info with player details
    final_df = pd.merge(
        roster_df,
        player_df,
        left_on='player_id',
        right_on='id',
        how='left'
    )
    
    # Select and reorder relevant columns
    columns_to_display = [
        'fullName',
        'position',
        'jerseyNumber',
        'status',
        'birthDate',
        'currentAge',
        'height',
        'weight',
        'birthCity',
        'birthCountry'
    ]
    
    # Only include columns that exist in the DataFrame
    final_columns = [col for col in columns_to_display if col in final_df.columns]
    
    return final_df[final_columns]

def get_all_teams() -> List[Dict]:
    """
    Get list of all MLB teams
    """
    url = 'https://statsapi.mlb.com/api/v1/teams?sportId=1'
    teams_data = call_mlb_api(url)
    return teams_data['teams']


def ensure_dataset_exists(client: bigquery.Client, dataset_id: str):
    """Create dataset if it doesn't exist"""
    project = client.project
    dataset_ref = f"{project}.{dataset_id}"
    
    try:
        client.get_dataset(dataset_ref)
        logger.info(f"Dataset {dataset_ref} already exists")
    except Exception as e:
        dataset = bigquery.Dataset(dataset_ref)
        dataset.location = "US"  # Specify the location
        dataset = client.create_dataset(dataset, timeout=30)
        logger.info(f"Created dataset {dataset_ref}")


def create_tables(dataset_id: str):
    """Create BigQuery tables if they don't exist"""
    ensure_dataset_exists(client, dataset_id)
    dataset_ref = client.dataset(dataset_id)
    
    # Create player profiles table
    player_table_ref = dataset_ref.table("player_profiles")
    player_table = bigquery.Table(player_table_ref, schema=PLAYER_PROFILE_SCHEMA)
    try:
        client.create_table(player_table)
        logger.info("Created player_profiles table")
    except Exception as e:
        logger.info(f"player_profiles table already exists: {str(e)}")
    
    # Create games table
    games_table_ref = dataset_ref.table("games")
    games_table = bigquery.Table(games_table_ref, schema=GAME_DATA_SCHEMA)
    try:
        client.create_table(games_table)
        logger.info("Created games table")
    except Exception as e:
        logger.info(f"games table already exists: {str(e)}")

def load_player_profiles(dataset_id: str):
    """Load player profiles into BigQuery"""
    # Get all teams
    teams = get_all_teams()
    all_players = []
    
    for team in teams:
        team_id = team['id']
        team_name = team['name']
        logger.info(f"Processing team: {team_name}")
        
        try:
          roster = get_team_roster(team_id)
          if roster.empty:
            logger.warning(f"Empty roster for team {team_name}")
            continue
          
          # Debug log to see roster structure
          logger.debug(f"Roster columns: {roster.columns}")
          logger.debug(f"First player data: {roster.iloc[0].to_dict()}")

          for _, player in roster.iterrows():
            # Convert birth_date string to proper date format
            birth_date = player.get('birthDate')
            if birth_date:
              
              try:
              # Parse the date string into a datetime object
               birth_date = pd.to_datetime(birth_date).strftime('%Y-%m-%d')
              except (ValueError, TypeError):
                   birth_date = None
            # Format the last_updated timestamp properly
            player_data = {
                'player_id': player.get('id'),
                'team_id': team_id,
                'team_name': team_name,
                'full_name': player.get('fullName'),
                'first_name': player.get('firstName'),
                'last_name': player.get('lastName'),
                'primary_position': player.get('position'),
                'jersey_number': player.get('jerseyNumber'),
                'birth_date': birth_date,
                'age': player.get('currentAge'),
                'height': player.get('height'),
                'weight': player.get('weight'),
                'birth_city': player.get('birthCity'),
                'birth_country': player.get('birthCountry'),
                'status': player.get('status'),
                'active': player.get('active', True),
                'last_updated': datetime.now(UTC)
            }
            

            # Verify required fields before adding
            if player_data['player_id'] and player_data['team_id']:
                    all_players.append(player_data)
            else:
                    logger.warning(f"Skipping player due to missing required fields: {player_data}")

        except Exception as e:
            logger.error(f"Error processing team {team_name}: {str(e)}")
            continue
    
    if not all_players:
        logger.error("No valid player data collected")
        return
    
    # Load to BigQuery
    table_id = f"{dataset_id}.player_profiles"
    job_config = bigquery.LoadJobConfig(
        write_disposition="WRITE_TRUNCATE",
        schema=PLAYER_PROFILE_SCHEMA
    )
    
    try:
      df = pd.DataFrame(all_players)

      # Print debug information
      logger.debug("DataFrame before loading:")
      logger.debug(f"Shape: {df.shape}")
      logger.debug("Columns and their null counts:")
      logger.debug(df.isnull().sum())

    # Convert date columns to datetime
      if 'birth_date' in df.columns:
         df['birth_date'] = pd.to_datetime(df['birth_date'])
      job = client.load_table_from_dataframe(df, table_id, job_config=job_config)
      job.result()
      logger.info(f"Loaded {len(all_players)} player profiles")
    except Exception as e:
        logger.error(f"Error loading data to BigQuery: {str(e)}")
        # Print the first few rows of the dataframe for debugging
        logger.debug("DataFrame head:")
        logger.debug(df.head())
        logger.debug("DataFrame dtypes:")
        logger.debug(df.dtypes)
        raise


def load_game_data(dataset_id: str, season: int = 2025):
    """Load game data into BigQuery"""
    schedule_url = f'https://statsapi.mlb.com/api/v1/schedule?sportId=1&season={season}'
    schedule_data = call_mlb_api(schedule_url)
    
    games = []
    for date in schedule_data.get('dates', []):
        for game in date.get('games', []):
            # Convert date strings to proper datetime objects
            game_date_str = game.get("gameDate")
            try:
                game_date = pd.to_datetime(game_date_str)
            except:
                game_date = None
            game_data = {
                'game_id': game.get('gamePk'),
                'game_date': game_date,
                'season': season,
                'game_type': game.get('gameType'),
                'status': game.get('status', {}).get('detailedState'),
                'home_team_id': game.get('teams', {}).get('home', {}).get('team', {}).get('id'),
                'away_team_id': game.get('teams', {}).get('away', {}).get('team', {}).get('id'),
                'home_team_name': game.get('teams', {}).get('home', {}).get('team', {}).get('name'),
                'away_team_name': game.get('teams', {}).get('away', {}).get('team', {}).get('name'),
                'home_score': game.get('teams', {}).get('home', {}).get('score'),
                'away_score': game.get('teams', {}).get('away', {}).get('score'),
                'venue_name': game.get('venue', {}).get('name'),
                'venue_id': game.get('venue', {}).get('id'),
                'last_updated': datetime.now(UTC)
            }
            games.append(game_data)
    
    # Load to BigQuery
    table_id = f"{dataset_id}.games"
    job_config = bigquery.LoadJobConfig(
        write_disposition="WRITE_APPEND",
        schema=GAME_DATA_SCHEMA
    )
    
    df = pd.DataFrame(games)
    job = client.load_table_from_dataframe(df, table_id, job_config=job_config)
    job.result()
    logger.info(f"Loaded {len(games)} games")

if __name__ == "__main__":
    # Set your dataset ID
    DATASET_ID = "mlb_data_2024"  # Replace with your dataset ID
    
    # Create tables
    create_tables(DATASET_ID)
    
    # Load data
    #load_player_profiles(DATASET_ID)
    load_game_data(DATASET_ID)