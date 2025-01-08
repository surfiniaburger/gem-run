import requests
import pandas as pd
from ratelimit import limits, sleep_and_retry
import logging
from typing import Dict, List
import os
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

CALLS = 100
RATE_LIMIT = 60

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
    
    Parameters:
    - df: pandas DataFrame containing player data
    - missing_threshold: maximum acceptable proportion of missing values (default 10%)
    
    Returns:
    - Cleaned DataFrame
    """
    # Calculate proportion of missing values for each column
    missing_proportions = df.isnull().sum() / len(df)
    
    # Identify columns to keep (those with missing values below threshold)
    columns_to_keep = missing_proportions[missing_proportions < missing_threshold].index
    
    # Log removed columns
    removed_columns = set(df.columns) - set(columns_to_keep)
    if removed_columns:
        logger.info(f"Removing columns due to missing values: {removed_columns}")
    
    # Return DataFrame with only the kept columns
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
    numeric_columns = ['jerseyNumber', 'currentAge', 'weight']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

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
            'jerseyNumber': player.get('jerseyNumber', None)
        }
        roster_list.append(player_info)
    
    roster_df = pd.DataFrame(roster_list)
    
    # Get detailed player info with error handling for each player
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
    
    # Select initial columns
    columns_to_display = [
        'player_id',
        'fullName',
        'position',
        'jerseyNumber',
        'status',
        'birthDate',
        'currentAge',
        'height',
        'weight',
        'birthCity',
        'birthCountry',
        'batSide.description',
        'pitchHand.description',
        'mlbDebutDate'
    ]
    
    # Only include columns that exist in the DataFrame
    final_columns = [col for col in columns_to_display if col in final_df.columns]
    final_df = final_df[final_columns]
    
    # Validate the data
    final_df = validate_player_data(final_df)
    
    # Clean the data by removing columns with too many missing values
    final_df = clean_player_data(final_df, missing_threshold=0.1)
    
    return final_df

def save_roster_to_csv(df: pd.DataFrame, team_name: str):
    """
    Save roster to CSV with timestamp and data quality report
    """
    # Create output directory if it doesn't exist
    output_dir = 'mlb_rosters'
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save main roster file
    filename = f"{output_dir}/{team_name}_roster_{timestamp}.csv"
    df.to_csv(filename, index=False)
    logger.info(f"Roster saved to {filename}")
    
    # Generate data quality report
    report_filename = f"{output_dir}/{team_name}_data_quality_{timestamp}.txt"
    with open(report_filename, 'w') as f:
        f.write(f"Data Quality Report for {team_name} Roster\n")
        f.write(f"Generated on: {datetime.now()}\n\n")
        f.write(f"Total players: {len(df)}\n")
        f.write(f"Columns in final dataset:\n{', '.join(df.columns)}\n\n")
        f.write(f"Missing values:\n{df.isnull().sum().to_string()}\n\n")
        f.write(f"Data types:\n{df.dtypes.to_string()}\n")
    
    logger.info(f"Data quality report saved to {report_filename}")
    
    return filename, report_filename

if __name__ == "__main__":
    logger.info("Fetching Dodgers roster...")
    dodgers_roster = get_team_roster(119)  # 119 is Dodgers team ID
    
    if not dodgers_roster.empty:
        # Save roster and get filenames
        roster_file, report_filename = save_roster_to_csv(dodgers_roster, "Dodgers")
        
        # Display preview of the data
        print("\nDodgers Roster Preview (first 5 rows):")
        print(dodgers_roster.head())
        print(f"\nFull roster saved to: {roster_file}")
        print(f"Data quality report saved to: {report_filename}")
    else:
        logger.error("Failed to retrieve Dodgers roster data")