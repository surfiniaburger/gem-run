import pandas as pd
import requests
import json
from datetime import datetime, UTC, timedelta
import logging
from typing import Dict, List, Tuple
from ratelimit import limits, sleep_and_retry
from google.cloud import bigquery
from google.api_core.exceptions import BadRequest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# BigQuery client setup
client = bigquery.Client()

CALLS = 100
RATE_LIMIT = 60

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

# Schema definitions (same as before)
def get_schema(team_name: str) -> List[bigquery.SchemaField]:
    """Generate schema with team-specific win/margin fields"""
    return [
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
        bigquery.SchemaField(f"{team_name}_win", "BOOLEAN"),
        bigquery.SchemaField(f"{team_name}_margin", "INTEGER"),
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

def create_tables(dataset_id: str, team_name: str):
    """Create BigQuery tables if they don't exist"""
    ensure_dataset_exists(client, dataset_id)
    dataset_ref = client.dataset(dataset_id)

    tables = {
        "games": get_schema(team_name),
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

def get_team_games(team_name: str, team_id: int, season: int = 2024) -> pd.DataFrame:
    """Fetch all games for specified team and season"""
    url = f'https://statsapi.mlb.com/api/v1/schedule?sportId=1&season={season}&teamId={team_id}'
    schedule_data = call_mlb_api(url)

    games_list = []
    for date in schedule_data.get('dates', []):
        for game in date.get('games', []):
            game_info = {
                'game_id': game['gamePk'],
                'official_date': pd.to_datetime(game['officialDate']).date(),
                'season': season,
                'game_type': game['gameType'],
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

            is_home = game_info['home_team_id'] == team_id
            team_score = game_info['home_score'] if is_home else game_info['away_score']
            opponent_score = game_info['away_score'] if is_home else game_info['home_score']

            game_info.update({
                f'{team_name}_win': team_score > opponent_score if game_info['status'] == 'Final' else None,
                f'{team_name}_margin': team_score - opponent_score if game_info['status'] == 'Final' else None
            })

            games_list.append(game_info)

    return pd.DataFrame(games_list)

def process_game_plays(game_pk: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Process play-by-play data for a specific game, with data validation."""
    url = f'https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live'
    game_data = call_mlb_api(url)

    plays_list = []
    player_stats = {'batters': {}, 'pitchers': {}}

    all_plays = game_data['liveData']['plays']['allPlays']
    for play in all_plays:
        # --- Data Validation: Check for critical missing data ---
        if not all([play.get('about'), play.get('matchup'), play.get('result'), play.get('count')]):
            logger.warning(f"Skipping play in game {game_pk} due to missing data: {play}")
            continue  # Skip this play and go to the next

        # Convert timestamps to datetime objects
        try:
            start_time = pd.to_datetime(play['about']['startTime'], utc=True)
            end_time = pd.to_datetime(play['about']['endTime'], utc=True)
        except (KeyError, ValueError) as e:
            logger.warning(f"Invalid or missing timestamp in game {game_pk}, play: {play}. Error: {e}")
            continue # Skip this play

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

        # Update player statistics (same logic as before, but with added checks)
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

    # --- Data Validation: Check for date consistency ---
    if not plays_df.empty:
        game_date_str = game_data['gameData']['datetime']['officialDate']
        game_date = pd.to_datetime(game_date_str, utc=True).date()
        if not all(plays_df['start_time'].dt.date == game_date):
            logger.warning(f"Date inconsistency detected in game {game_pk}. Plays found outside of game date: {game_date_str}")
            #  Decide on a course of action:
            #  1. Skip the entire game:
            #     return pd.DataFrame(), pd.DataFrame()
            #  2. Filter out inconsistent plays:
            #     plays_df = plays_df[plays_df['start_time'].dt.date == game_date]
            #  3. Correct the plays' start_time (risky, might introduce errors):
            #     plays_df['start_time'] = plays_df['start_time'].apply(lambda x: x.replace(year=game_date.year, month=game_date.month, day=game_date.day))
            # For this example, let's filter out the inconsistent plays, as it's the safest option:
            plays_df = plays_df[plays_df['start_time'].dt.date == game_date]

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
def upload_to_bigquery(df: pd.DataFrame, table_id: str, schema: List[bigquery.SchemaField]):
    """Upload DataFrame to BigQuery table, handling BigQuery errors."""
    job_config = bigquery.LoadJobConfig(
        write_disposition="WRITE_APPEND",  # Append to existing data
        schema=schema
    )

    try:
        job = client.load_table_from_dataframe(df, table_id, job_config=job_config)
        job.result()  # Wait for the job to complete
        logger.info(f"Loaded {len(df)} rows to {table_id}")

    except BadRequest as e:
        logger.error(f"BigQuery BadRequest: {e}")
        for error in e.errors:
            logger.error(f"  Reason: {error['reason']}, Message: {error['message']}")
        raise  # Re-raise the exception to halt execution if desired

    except Exception as e:
        logger.error(f"Error loading data to {table_id}: {e}")
        raise

def process_team_data(team_name: str, team_id: int, n_games: int = 50):
    """Process recent games for a specific team"""
    dataset_id = f"{team_name}_mlb_data_2024"
    logger.info(f"Processing {team_name} data...")

    # Create tables with team-specific schema
    create_tables(dataset_id, team_name)

    # Get recent games
    games_df = get_team_games(team_name, team_id)
    recent_games = games_df.sort_values('official_date', ascending=False).head(n_games)

    # Upload games data
    upload_to_bigquery(
        recent_games,
        f"{dataset_id}.games",
        get_schema(team_name)
    )

    # Process and upload plays and player stats
    for game_id in recent_games['game_id']:
        try:
            plays_df, player_stats_df = process_game_plays(game_id)

            if not plays_df.empty:
                upload_to_bigquery(
                    plays_df,
                    f"{dataset_id}.plays",
                    PLAYS_SCHEMA
                )

            if not player_stats_df.empty:
                upload_to_bigquery(
                    player_stats_df,
                    f"{dataset_id}.player_stats",
                    PLAYER_STATS_SCHEMA
                )

        except Exception as e:
            logger.error(f"Error processing game {game_id}: {e}")
            continue

def main():
    """Process data for all teams"""
    for team_name, team_id in TEAMS.items():
        try:
            process_team_data(team_name, team_id)
            logger.info(f"Successfully completed {team_name} data processing")
        except Exception as e:
            logger.error(f"Error processing {team_name}: {e}")
            continue

if __name__ == "__main__":
    main()