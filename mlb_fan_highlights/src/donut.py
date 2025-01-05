import requests
import pandas as pd
from ratelimit import limits, sleep_and_retry
import logging
from typing import Dict, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Rate limiting decorators
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

def get_all_rosters(season: int = 2024) -> Dict[str, pd.DataFrame]:
    """
    Get rosters for all MLB teams
    """
    teams = get_all_teams()
    rosters = {}
    
    for team in teams:
        team_name = team['name']
        team_id = team['id']
        logger.info(f"Fetching roster for {team_name}")
        
        try:
            roster = get_team_roster(team_id, season)
            print(roster)
            if not roster.empty:
                rosters[team_name] = roster
        except Exception as e:
            logger.error(f"Error fetching roster for {team_name}: {str(e)}")
    
    return rosters

# Example usage
if __name__ == "__main__":
    # Get roster for a single team (e.g., Dodgers, ID 119)
    #print("Fetching Dodgers roster...")
    #dodgers_roster = get_team_roster(119)
    #print("\nDodgers Roster:")
    #print(dodgers_roster.head())
    
    # Uncomment to get all team rosters
     print("\nFetching all team rosters...")
     all_rosters = get_all_rosters()
     print("\nNumber of teams with rosters:", len(all_rosters))