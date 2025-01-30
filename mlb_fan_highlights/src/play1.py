#Existing Setup
import pandas as pd
import requests
import json
from datetime import datetime, UTC
import logging
from typing import Dict, List, Tuple
from ratelimit import limits, sleep_and_retry
from google.cloud import bigquery
from google.cloud import pubsub_v1
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# BigQuery client setup
client = bigquery.Client()

CALLS = 100
RATE_LIMIT = 60

# Schema definitions
GAMES_SCHEMA = [
    bigquery.SchemaField("game_id", "INTEGER", mode="REQUIRED"),
    bigquery.SchemaField("official_date", "DATE", mode="REQUIRED"),
    bigquery.SchemaField("game_type", "STRING"),
    bigquery.SchemaField("season", "INTEGER"),
    bigquery.SchemaField("home_team_id", "INTEGER"),
    bigquery.SchemaField("home_team_name", "STRING"),
    bigquery.SchemaField("home_score", "INTEGER"),
    bigquery.SchemaField("away_team_id", "INTEGER"),
    bigquery.SchemaField("away_team_name", "STRING"),
    bigquery.SchemaField("away_score", "INTEGER"),
    bigquery.SchemaField("venue_name", "STRING"),
    bigquery.SchemaField("status", "STRING"),
    bigquery.SchemaField("dodgers_win", "BOOLEAN"),
    bigquery.SchemaField("dodgers_margin", "INTEGER"),
    bigquery.SchemaField("last_updated", "TIMESTAMP")
]

PLAYS_SCHEMA = [
    bigquery.SchemaField("play_id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("game_id", "INTEGER", mode="REQUIRED"),
    bigquery.SchemaField("batter_id", "INTEGER"),
    bigquery.SchemaField("pitcher_id", "INTEGER"),
    bigquery.SchemaField("inning", "INTEGER"),
    bigquery.SchemaField("half_inning", "STRING"),
    bigquery.SchemaField("event", "STRING"),
    bigquery.SchemaField("event_type", "STRING"),
    bigquery.SchemaField("description", "STRING"),
    bigquery.SchemaField("balls", "INTEGER"),
    bigquery.SchemaField("strikes", "INTEGER"),
    bigquery.SchemaField("outs", "INTEGER"),
    bigquery.SchemaField("start_time", "TIMESTAMP"),
    bigquery.SchemaField("end_time", "TIMESTAMP"),
    bigquery.SchemaField("rbi", "INTEGER"),
    bigquery.SchemaField("is_scoring_play", "BOOLEAN"),
    bigquery.SchemaField("last_updated", "TIMESTAMP")
]

PLAYER_STATS_SCHEMA = [
    bigquery.SchemaField("player_id", "INTEGER", mode="REQUIRED"),
    bigquery.SchemaField("game_id", "INTEGER", mode="REQUIRED"),
    bigquery.SchemaField("at_bats", "INTEGER"),
    bigquery.SchemaField("hits", "INTEGER"),
    bigquery.SchemaField("singles", "INTEGER"),
    bigquery.SchemaField("doubles", "INTEGER"),
    bigquery.SchemaField("triples", "INTEGER"),
    bigquery.SchemaField("home_runs", "INTEGER"),
    bigquery.SchemaField("walks", "INTEGER"),
    bigquery.SchemaField("strikeouts", "INTEGER"),
    bigquery.SchemaField("rbi", "INTEGER"),
    bigquery.SchemaField("batting_average", "FLOAT"),
    bigquery.SchemaField("on_base_percentage", "FLOAT"),
    bigquery.SchemaField("slugging_percentage", "FLOAT"),
    bigquery.SchemaField("batters_faced", "INTEGER"),
    bigquery.SchemaField("strikes", "INTEGER"),
    bigquery.SchemaField("balls", "INTEGER"),
    bigquery.SchemaField("hits_allowed", "INTEGER"),
    bigquery.SchemaField("runs_allowed", "INTEGER"),
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

def create_tables(dataset_id: str):
    """Create BigQuery tables if they don't exist"""
    ensure_dataset_exists(client, dataset_id)
    dataset_ref = client.dataset(dataset_id)
    
    tables = {
        "games": GAMES_SCHEMA,
        "plays": PLAYS_SCHEMA,
        "player_stats": PLAYER_STATS_SCHEMA
    }
    
    for table_name, schema in tables.items():
        table_ref = dataset_ref.table(table_name)
        table = bigquery.Table(table_ref, schema=schema)
        try:
            client.create_table(table)
            logger.info(f"Created {table_name} table")
        except Exception as e:
            logger.info(f"{table_name} table already exists: {str(e)}")

def get_dodgers_games(season: int = 2024) -> pd.DataFrame:
    """Fetch all Dodgers games for the specified season"""
    url = f'https://statsapi.mlb.com/api/v1/schedule?sportId=1&season={season}&teamId=119'
    schedule_data = call_mlb_api(url)
    
    games_list = []
    for date in schedule_data.get('dates', []):
        for game in date.get('games', []):
            game_info = {
                'game_id': game['gamePk'],
                'official_date': pd.to_datetime(game['officialDate']).date(),
                'season': season,
                "game_type": game['gameType'],
                'home_team_id': game['teams']['home']['team']['id'],
                'home_team_name': game['teams']['home']['team']['name'],
                'home_score': game['teams']['home'].get('score', 0),
                'away_team_id': game['teams']['away']['team']['id'],
                'away_team_name': game['teams']['away']['team']['name'],
                'away_score': game['teams']['away'].get('score', 0),
                'venue_name': game['venue']['name'],
                'status': game['status']['detailedState'],
                'last_updated': datetime.now(UTC)
            }
            
            # Calculate Dodgers-specific fields
            is_home = game_info['home_team_id'] == 119
            dodgers_score = game_info['home_score'] if is_home else game_info['away_score']
            opponent_score = game_info['away_score'] if is_home else game_info['home_score']
            
            game_info.update({
                'dodgers_win': dodgers_score > opponent_score if game_info['status'] == 'Final' else None,
                'dodgers_margin': dodgers_score - opponent_score if game_info['status'] == 'Final' else None
            })
            
            games_list.append(game_info)
    
    return pd.DataFrame(games_list)

def process_game_plays(game_pk: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Process play-by-play data for a specific game"""
    url = f'https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live'
    game_data = call_mlb_api(url)
    
    plays_list = []
    player_stats = {'batters': {}, 'pitchers': {}}
    
    all_plays = game_data['liveData']['plays']['allPlays']
    for play in all_plays:
        # Convert timestamps to datetime objects
        try:
            start_time = pd.to_datetime(play['about']['startTime'], utc=True)
            end_time = pd.to_datetime(play['about']['endTime'], utc=True)
        except (KeyError, ValueError):
            # If timestamps are missing or invalid, use current time
            current_time = datetime.now(UTC)
            start_time = current_time
            end_time = current_time
        play_info = {
            'play_id': f"{game_pk}_{play['about']['atBatIndex']}",
            'game_id': game_pk,
            'batter_id': play['matchup']['batter']['id'],
            'pitcher_id': play['matchup']['pitcher']['id'],
            'inning': play['about']['inning'],
            'half_inning': play['about']['halfInning'],
            'event': play['result']['event'],
            'event_type': play['result']['eventType'],
            'description': play['result']['description'],
            'balls': play['count']['balls'],
            'strikes': play['count']['strikes'],
            'outs': play['count']['outs'],
            'start_time': start_time,
            'end_time': end_time,
            'rbi': play['result'].get('rbi', 0),
            'is_scoring_play': play['about']['isScoringPlay'],
            'last_updated': datetime.now(UTC)
        }
        plays_list.append(play_info)
        
        # Update player statistics (same logic as before)
        batter_id = play_info['batter_id']
        pitcher_id = play_info['pitcher_id']
        
        if batter_id not in player_stats['batters']:
            player_stats['batters'][batter_id] = {
                'player_id': batter_id,
                'game_id': game_pk,
                'at_bats': 0,
                'hits': 0,
                'singles': 0,
                'doubles': 0,
                'triples': 0,
                'home_runs': 0,
                'walks': 0,
                'strikeouts': 0,
                'rbi': 0
            }
            
        if pitcher_id not in player_stats['pitchers']:
            player_stats['pitchers'][pitcher_id] = {
                'player_id': pitcher_id,
                'game_id': game_pk,
                'batters_faced': 0,
                'strikes': 0,
                'balls': 0,
                'strikeouts': 0,
                'walks': 0,
                'hits_allowed': 0,
                'runs_allowed': 0
            }
        
        # Update stats based on event
        event = play_info['event']
        if event != 'Walk':
            player_stats['batters'][batter_id]['at_bats'] += 1
        
        if event == 'Single':
            player_stats['batters'][batter_id]['hits'] += 1
            player_stats['batters'][batter_id]['singles'] += 1
        elif event == 'Double':
            player_stats['batters'][batter_id]['hits'] += 1
            player_stats['batters'][batter_id]['doubles'] += 1
        elif event == 'Triple':
            player_stats['batters'][batter_id]['hits'] += 1
            player_stats['batters'][batter_id]['triples'] += 1
        elif event == 'Home Run':
            player_stats['batters'][batter_id]['hits'] += 1
            player_stats['batters'][batter_id]['home_runs'] += 1
        elif event == 'Strikeout':
            player_stats['batters'][batter_id]['strikeouts'] += 1
        elif event == 'Walk':
            player_stats['batters'][batter_id]['walks'] += 1
        
        player_stats['batters'][batter_id]['rbi'] += play_info['rbi']
        
        # Update pitching stats
        player_stats['pitchers'][pitcher_id]['batters_faced'] += 1
        player_stats['pitchers'][pitcher_id]['strikes'] += play_info['strikes']
        player_stats['pitchers'][pitcher_id]['balls'] += play_info['balls']
        
        if event == 'Strikeout':
            player_stats['pitchers'][pitcher_id]['strikeouts'] += 1
        elif event == 'Walk':
            player_stats['pitchers'][pitcher_id]['walks'] += 1
        elif event in ['Single', 'Double', 'Triple', 'Home Run']:
            player_stats['pitchers'][pitcher_id]['hits_allowed'] += 1
        
        if play_info['is_scoring_play']:
            player_stats['pitchers'][pitcher_id]['runs_allowed'] += play_info['rbi']
    
    # Convert to DataFrames and calculate additional metrics
    plays_df = pd.DataFrame(plays_list)

    # Ensure timestamp columns are properly formatted
    for col in ['start_time', 'end_time', 'last_updated']:
        plays_df[col] = pd.to_datetime(plays_df[col], utc=True)
    
    # Process player stats
    all_stats = []
    for stats_dict in [player_stats['batters'], player_stats['pitchers']]:
        for player_stats in stats_dict.values():
            player_stats['last_updated'] = datetime.now(UTC)
            all_stats.append(player_stats)
    
    player_stats_df = pd.DataFrame(all_stats)
    
    # Calculate batting metrics where applicable
    if not player_stats_df.empty and 'at_bats' in player_stats_df.columns:
        player_stats_df['batting_average'] = player_stats_df['hits'] / player_stats_df['at_bats']
        player_stats_df['on_base_percentage'] = (player_stats_df['hits'] + player_stats_df['walks']) / \
                                              (player_stats_df['at_bats'] + player_stats_df['walks'])
        player_stats_df['slugging_percentage'] = (player_stats_df['singles'] + 
                                                2 * player_stats_df['doubles'] + 
                                                3 * player_stats_df['triples'] + 
                                                4 * player_stats_df['home_runs']) / \
                                               player_stats_df['at_bats']
    # Ensure last_updated is properly formatted
    player_stats_df['last_updated'] = pd.to_datetime(player_stats_df['last_updated'], utc=True)    
    
    return plays_df, player_stats_df

def publish_to_pubsub(data: dict, topic_id: str, project_id: str):
    """Publishes data to Pub/Sub topic."""
    from datetime import date, datetime
    import pandas as pd
    
    publisher = pubsub_v1.PublisherClient()
    topic_path = publisher.topic_path(project_id, topic_id)
    
    processed_data = {}
    for key, value in data.items():
        if isinstance(value, (datetime, pd.Timestamp)):
            processed_data[key] = value.isoformat()
        elif isinstance(value, pd.DatetimeTZDtype):
            processed_data[key] = value.isoformat()
        elif isinstance(value, date):
            processed_data[key] = value.isoformat()
        else:
            processed_data[key] = value
    
    try:
        data_str = json.dumps(processed_data).encode("utf-8")
        future = publisher.publish(topic_path, data=data_str)
        future.result()
        logger.info(f"Published message to {topic_path}.")
    except Exception as e:
        logger.error(f"Error publishing message to Pub/Sub: {e}")
        raise

def process_recent_games(dataset_id: str, topic_id: str, project_id: str, n_games: int = 5):
    """Process recent Dodgers games and upload to Pub/Sub"""
    # Create tables if they don't exist
    create_tables(dataset_id)
    
    # Get recent games
    games_df = get_dodgers_games()
    recent_games = games_df.sort_values('official_date', ascending=False).head(n_games)
    
    # Publish games data
    for _, row in recent_games.iterrows():
        publish_to_pubsub(row.to_dict(), topic_id, project_id)
    
    # Process and publish plays and player stats
    for game_id in recent_games['game_id']:
        try:
            plays_df, player_stats_df = process_game_plays(game_id)
            
            # Publish plays data
            if not plays_df.empty:
              for _, row in plays_df.iterrows():
                publish_to_pubsub(row.to_dict(), topic_id, project_id)
            
            # Publish player stats data
            if not player_stats_df.empty:
              for _, row in player_stats_df.iterrows():
                publish_to_pubsub(row.to_dict(), topic_id, project_id)
                
        except Exception as e:
            logger.error(f"Error processing game {game_id}: {str(e)}")
            continue
            
def create_topic_if_not_exists(project_id: str, topic_id: str):
    """Creates a Pub/Sub topic if it doesn't exist"""
    publisher = pubsub_v1.PublisherClient()
    topic_path = publisher.topic_path(project_id, topic_id)
    
    try:
        publisher.create_topic(request={"name": topic_path})
        logger.info(f"Created topic: {topic_path}")
    except Exception as e:
        if 'AlreadyExists' in str(e):
            logger.info(f"Topic {topic_path} already exists")
        else:
            raise

def main():
    DATASET_ID = "dodgers_mlb_data_2024"
    PROJECT_ID = "gem-rush-007"
    TOPIC_ID = "mlb-data-topic"
    
    if not PROJECT_ID:
        raise ValueError("PROJECT_ID environment variable must be set")

    logger.info("Starting Dodgers data processing...")
    
    try:
        #create_topic_if_not_exists(PROJECT_ID, TOPIC_ID)
        process_recent_games(DATASET_ID, TOPIC_ID, PROJECT_ID, n_games=91)
        logger.info("Successfully completed data processing and upload")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()