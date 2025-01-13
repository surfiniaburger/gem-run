import pandas as pd
import requests
import json
from datetime import datetime, UTC
import logging
from typing import Dict, List, Tuple
from ratelimit import limits, sleep_and_retry
from google.cloud import bigquery

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# BigQuery client setup
client = bigquery.Client()

CALLS = 100
RATE_LIMIT = 60


# Add this schema definition at the top with other schemas

TEAMS_SCHEMA = [
    bigquery.SchemaField("team_id", "INTEGER", mode="REQUIRED"),
    bigquery.SchemaField("name", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("team_code", "STRING"),
    bigquery.SchemaField("file_code", "STRING"),
    bigquery.SchemaField("abbreviation", "STRING"),
    bigquery.SchemaField("team_name", "STRING"),
    bigquery.SchemaField("location_name", "STRING"),
    bigquery.SchemaField("league_id", "INTEGER"),
    bigquery.SchemaField("league_name", "STRING"),
    bigquery.SchemaField("division_id", "INTEGER"),
    bigquery.SchemaField("division_name", "STRING"),
    bigquery.SchemaField("venue_id", "INTEGER"),
    bigquery.SchemaField("venue_name", "STRING"),
    bigquery.SchemaField("spring_venue_id", "INTEGER"),
    bigquery.SchemaField("spring_venue_name", "STRING"),
    bigquery.SchemaField("active", "BOOLEAN"),
    bigquery.SchemaField("season", "INTEGER"),
    bigquery.SchemaField("first_year_of_play", "STRING"),
    bigquery.SchemaField("last_updated", "TIMESTAMP")
]

@sleep_and_retry
@limits(calls=CALLS, period=RATE_LIMIT)
def call_mlb_api(url: str) -> Dict:
    """Make a rate-limited call to the MLB API"""
    response = requests.get(url)
    response.raise_for_status()
    return response.json()


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


def get_mlb_teams(season: int = 2024) -> pd.DataFrame:
    """
    Fetch all MLB teams for the specified season
    """
    url = f'https://statsapi.mlb.com/api/v1/teams?sportId=1&season={season}'
    teams_data = call_mlb_api(url)
    
    teams_list = []
    for team in teams_data.get('teams', []):
        team_info = {
            'team_id': team['id'],
            'name': team['name'],
            'team_code': team.get('teamCode', ''),
            'file_code': team.get('fileCode', ''),
            'abbreviation': team.get('abbreviation', ''),
            'team_name': team.get('teamName', ''),
            'location_name': team.get('locationName', ''),
            'league_id': team.get('league', {}).get('id'),
            'league_name': team.get('league', {}).get('name'),
            'division_id': team.get('division', {}).get('id'),
            'division_name': team.get('division', {}).get('name'),
            'venue_id': team.get('venue', {}).get('id'),
            'venue_name': team.get('venue', {}).get('name'),
            'spring_venue_id': team.get('springVenue', {}).get('id'),
            'spring_venue_name': team.get('springVenue', {}).get('name'),
            'active': team.get('active', True),
            'season': season,
            'first_year_of_play': team.get('firstYearOfPlay'),
            'last_updated': datetime.now(UTC)
        }
        teams_list.append(team_info)
    
    return pd.DataFrame(teams_list)


def upload_to_bigquery(df: pd.DataFrame, table_id: str, schema: List[bigquery.SchemaField]):
    """Upload DataFrame to BigQuery table"""
    job_config = bigquery.LoadJobConfig(
        write_disposition="WRITE_APPEND",
        schema=schema
    )
    
    try:
        job = client.load_table_from_dataframe(df, table_id, job_config=job_config)
        job.result()
        logger.info(f"Loaded {len(df)} rows to {table_id}")
    except Exception as e:
        logger.error(f"Error loading data to {table_id}: {str(e)}")
        raise


def process_teams(dataset_id: str):
    """
    Process MLB teams data and upload to BigQuery
    """
    # Ensure teams table exists
    ensure_dataset_exists(client, dataset_id)
    dataset_ref = client.dataset(dataset_id)
    table_ref = dataset_ref.table('teams')
    table = bigquery.Table(table_ref, schema=TEAMS_SCHEMA)
    
    try:
        client.create_table(table)
        logger.info("Created teams table")
    except Exception as e:
        logger.info(f"Teams table already exists: {str(e)}")
    
    # Get teams data
    teams_df = get_mlb_teams()
    
    # Upload teams data
    upload_to_bigquery(
        teams_df,
        f"{dataset_id}.teams",
        TEAMS_SCHEMA
    )
    logger.info(f"Uploaded {len(teams_df)} teams to BigQuery")

# Modify the main function to include teams processing
def main():
    """Main execution function"""
    DATASET_ID = "dodgers_mlb_data_2024"
    
    logger.info("Starting MLB data processing...")
    
    try:
        # Process teams data
        logger.info("Processing teams data...")
        process_teams(DATASET_ID)
        logger.info("Successfully completed data processing and upload")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()