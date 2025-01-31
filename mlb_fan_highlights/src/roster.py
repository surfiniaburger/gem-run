import requests
import pandas as pd
from ratelimit import limits, sleep_and_retry
import logging
from typing import Dict, List
from datetime import datetime, UTC
from google.cloud import bigquery

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# BigQuery client setup
client = bigquery.Client()

CALLS = 100
RATE_LIMIT = 60

# Schema definition for roster table
ROSTER_SCHEMA = [
    bigquery.SchemaField("player_id", "INTEGER", mode="REQUIRED"),
    bigquery.SchemaField("full_name", "STRING"),
    bigquery.SchemaField("position", "STRING"),
    bigquery.SchemaField("jersey_number", "INTEGER"),
    bigquery.SchemaField("status", "STRING"),
    bigquery.SchemaField("birth_date", "DATE"),
    bigquery.SchemaField("current_age", "INTEGER"),
    bigquery.SchemaField("height", "STRING"),
    bigquery.SchemaField("weight", "INTEGER"),
    bigquery.SchemaField("birth_city", "STRING"),
    bigquery.SchemaField("birth_country", "STRING"),
    bigquery.SchemaField("bat_side", "STRING"),
    bigquery.SchemaField("pitch_hand", "STRING"),
    bigquery.SchemaField("mlb_debut_date", "DATE"),
    bigquery.SchemaField("last_updated", "TIMESTAMP")
]


# Team configurations
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


@sleep_and_retry
@limits(calls=CALLS, period=RATE_LIMIT)
def call_mlb_api(url: str) -> Dict:
    """
    Make a rate-limited call to the MLB API
    """
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

def clean_player_data(df: pd.DataFrame, missing_threshold: float = 0.1) -> pd.DataFrame:
    """
    Clean player data by removing columns with too many missing values
    """
    # Calculate proportion of missing values for each column
    missing_proportions = df.isnull().sum() / len(df)
    
    # Identify columns to keep (those with missing values below threshold)
    columns_to_keep = missing_proportions[missing_proportions < missing_threshold].index
    
    # Log removed columns
    removed_columns = set(df.columns) - set(columns_to_keep)
    if removed_columns:
        logger.info(f"Removing columns due to missing values: {removed_columns}")
    
    return df[columns_to_keep]

def validate_player_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and clean player data
    """
    # Check for missing values
    missing_values = df.isnull().sum()
    if missing_values.any():
        logger.info(f"Missing values before cleaning:\n{missing_values[missing_values > 0]}")
    
    # Check for duplicate player IDs
    if 'player_id' in df.columns:
        duplicates = df[df.duplicated(['player_id'], keep=False)]
        if not duplicates.empty:
            logger.warning(f"Duplicate player IDs found:\n{duplicates[['player_id', 'fullName']]}")
    
    # Ensure numeric types for relevant columns
    numeric_columns = ['jersey_number', 'current_age', 'weight']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Convert date columns
    date_columns = ['birth_date', 'mlb_debut_date']
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col]).dt.date
    
    return df

def ensure_dataset_exists(client: bigquery.Client, dataset_id: str):
    """Create dataset if it doesn't exist"""
    project = client.project
    dataset_ref = f"{project}.{dataset_id}"
    
    try:
        client.get_dataset(dataset_ref)
        logger.info(f"Dataset {dataset_ref} already exists")
    except Exception:
        dataset = bigquery.Dataset(dataset_ref)
        dataset.location = "US"
        dataset = client.create_dataset(dataset, timeout=30)
        logger.info(f"Created dataset {dataset_ref}")

def create_roster_table(dataset_id: str):
    """Create BigQuery roster table if it doesn't exist"""
    ensure_dataset_exists(client, dataset_id)
    dataset_ref = client.dataset(dataset_id)
    
    table_ref = dataset_ref.table("roster")
    table = bigquery.Table(table_ref, schema=ROSTER_SCHEMA)
    try:
        client.create_table(table)
        logger.info("Created roster table")
    except Exception as e:
        logger.info(f"Roster table already exists: {str(e)}")

def upload_to_bigquery(df: pd.DataFrame, table_id: str):
    """Upload DataFrame to BigQuery table"""
    job_config = bigquery.LoadJobConfig(
        write_disposition="WRITE_TRUNCATE",  # Overwrite existing data
        schema=ROSTER_SCHEMA
    )
    
    try:
        job = client.load_table_from_dataframe(df, table_id, job_config=job_config)
        job.result()
        logger.info(f"Loaded {len(df)} rows to {table_id}")
    except Exception as e:
        logger.error(f"Error loading data to {table_id}: {str(e)}")
        raise

def get_team_roster(team_id: int, season: int = 2024) -> pd.DataFrame:
    """
    Get roster for a specific team and season with enhanced error handling and validation
    """
    url = f'https://statsapi.mlb.com/api/v1/teams/{team_id}/roster?season={season}'
    try:
        roster_data = call_mlb_api(url)
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch roster data: {e}")
        return pd.DataFrame()
    
    if 'roster' not in roster_data:
        logger.warning(f"No roster data found for team {team_id}")
        return pd.DataFrame()
    
    # Extract player IDs and basic roster info
    roster_list = []
    for player in roster_data['roster']:
        player_info = {
            'player_id': player['person']['id'],
            'position': player['position']['name'],
            'status': player.get('status', {}).get('description', 'Unknown'),
            'jersey_number': player.get('jerseyNumber', None)
        }
        roster_list.append(player_info)
    
    roster_df = pd.DataFrame(roster_list)
    
    # Get detailed player info
    player_details = []
    for player_id in roster_df['player_id']:
        try:
            player_url = f"https://statsapi.mlb.com/api/v1/people/{player_id}"
            player_data = call_mlb_api(player_url)
            if 'people' in player_data and len(player_data['people']) > 0:
                player_details.append(player_data['people'][0])
            else:
                logger.warning(f"No detailed data found for player ID {player_id}")
        except Exception as e:
            logger.error(f"Error fetching details for player ID {player_id}: {e}")
    
    if not player_details:
        logger.error("No player details were retrieved")
        return pd.DataFrame()
    
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
    
    # Rename columns to match schema
    column_mapping = {
        'fullName': 'full_name',
        'jerseyNumber': 'jersey_number',
        'birthDate': 'birth_date',
        'currentAge': 'current_age',
        'birthCity': 'birth_city',
        'birthCountry': 'birth_country',
        'batSide.description': 'bat_side',
        'pitchHand.description': 'pitch_hand',
        'mlbDebutDate': 'mlb_debut_date'
    }
    final_df = final_df.rename(columns=column_mapping)

    # Add last_updated timestamp before selecting schema columns
    final_df['last_updated'] = datetime.now(UTC)
    
    # Select columns that match schema
    schema_columns = [field.name for field in ROSTER_SCHEMA]
    final_df = final_df[schema_columns]
    
    # Add last_updated timestamp
    final_df['last_updated'] = datetime.now(UTC)
    
    # Validate and clean the data
    final_df = validate_player_data(final_df)
    final_df = clean_player_data(final_df, missing_threshold=0.1)
    
    return final_df



def main():
    """Main execution function"""
    
    logger.info("Starting MLB roster processing for all teams...")
    
    for team_name, team_id in TEAMS.items():
        DATASET_ID = f"{team_name.lower().replace(' ', '_')}_mlb_data_2024"
        logger.info(f"Processing roster for {team_name}...")
        
        try:
            # Create table if it doesn't exist
            create_roster_table(DATASET_ID)
            
            # Get team roster
            team_roster = get_team_roster(team_id)  
            
            if not team_roster.empty:
                # Upload to BigQuery
                table_id = f"{DATASET_ID}.roster"
                upload_to_bigquery(team_roster, table_id)
                
                # Display preview of the data
                print(f"\n{team_name} Roster Preview (first 5 rows):")
                print(team_roster.head())
                print(f"Roster data uploaded to BigQuery table: {table_id}")
            else:
                logger.error(f"Failed to retrieve roster data for {team_name}")
        except Exception as e:
            logger.error(f"Error processing roster for {team_name}: {str(e)}")

if __name__ == "__main__":
    main()