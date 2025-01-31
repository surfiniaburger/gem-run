from IPython.display import HTML, Markdown, display
from google import genai
from google.genai.types import (
    GenerateContentConfig,
    GoogleSearch,
    Part,
    Retrieval,
    SafetySetting,
    Tool,
    VertexAISearch,
)
from google.cloud import bigquery
import os
import logging
from google.api_core import exceptions
from typing import List, Dict, Union, Optional
from datetime import datetime
import urllib.parse
import json


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


PROJECT_ID = "gem-rush-007"  # Replace with your actual Google Cloud project ID
os.environ["GOOGLE_CLOUD_PROJECT"] = PROJECT_ID # set this environment variable to your project ID
bq_client = bigquery.Client(project=PROJECT_ID)

client = genai.Client(vertexai=True, project="gem-rush-007", location="us-central1")
MODEL_ID = "gemini-2.0-flash-exp"  # @param {type: "string"}

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

# Dictionary to map full team names to their corresponding keys in TEAMS
FULL_TEAM_NAMES = {
    'texas rangers': 'rangers',
    'los angeles angels': 'angels',
    'houston astros': 'astros',
    'tampa bay rays': 'rays',
    'toronto blue jays': 'blue_jays',
    'new york yankees': 'yankees',
    'baltimore orioles': 'orioles',
    'boston red sox': 'red_sox',
    'minnesota twins': 'twins',
    'chicago white sox': 'white_sox',
    'cleveland guardians': 'guardians',
    'detroit tigers': 'tigers',
    'kansas city royals': 'royals',
    'san diego padres': 'padres',
    'san francisco giants': 'giants',
    'arizona diamondbacks': 'diamondbacks',
    'colorado rockies': 'rockies',
    'philadelphia phillies': 'phillies',
    'atlanta braves': 'braves',
    'miami marlins': 'marlins',
    'washington nationals': 'nationals',
    'new york mets': 'mets',
    'pittsburgh pirates': 'pirates',
    'st louis cardinals': 'cardinals',
    'milwaukee brewers': 'brewers',
    'chicago cubs': 'cubs',
    'cincinnati reds': 'reds',
    'oakland athletics': 'athletics',
    'seattle mariners': 'mariners',
    'los angeles dodgers': 'dodgers',
}

def _get_table_name(team_name: str) -> str:
    """
    Helper function to construct the table name from a team's full name.
    
    Args:
        team_name (str): The full team name (e.g., "Minnesota Twins", "Arizona Diamondbacks")
    
    Returns:
        str: The formatted table name (e.g., "`gem-rush-007.twins_mlb_data_2024`")
    """
    # Convert to lowercase for consistent matching
    cleaned_name = team_name.lower().strip()
    
    # Try to find the team in the full names mapping
    if cleaned_name in FULL_TEAM_NAMES:
        team_key = FULL_TEAM_NAMES[cleaned_name]
        return f"`gem-rush-007.{team_key}_mlb_data_2024`"
    
    # If the exact full name isn't found, try to match with the team key directly
    for team_key in TEAMS:
        if team_key in cleaned_name:
            return f"`gem-rush-007.{team_key}_mlb_data_2024`"
    
    # If no match is found, return unknown table name
    return f"`gem-rush-007.unknown_team_mlb_data_2024`"


def fetch_team_games(team_name: str) -> list:
    """
    Fetches all games (both home and away) for a specified team with detailed game information.

    Args:
        team_name (str): The team name as it appears in the TEAMS dictionary

    Returns:
        list: A list of dictionaries containing game details including:
              - game_id
              - official_date (in YYYY-MM-DD format)
              - home/away team IDs and names
              - game_type
              - scores
              - venue
              - game status
              - team win/loss and margin

    Raises:
        ValueError: If team_name is not found in TEAMS dictionary
        BigQueryError: If there's an issue with the BigQuery execution
        Exception: For other unexpected errors
    """
    
    team_id = TEAMS[team_name]
    table_name = _get_table_name(team_name)
    
    try:
        query = f"""
        SELECT
            game_id,
            official_date,
            home_team_id,
            game_type,
            home_team_name,
            away_team_id,
            away_team_name,
            home_score,
            away_score,
            venue_name,
            status,
            {team_name}_win as team_win,
            {team_name}_margin as team_margin
        FROM
            {table_name}.games
        WHERE
            (home_team_id = {team_id} OR away_team_id = {team_id})
        """
        
        bq_client = bigquery.Client()
        query_job = bq_client.query(query)
        
        results = []
        for row in query_job:
            row_dict = dict(row)
            
            if row_dict.get('official_date'):
                row_dict['official_date'] = row_dict['official_date'].strftime('%Y-%m-%d')
            
            required_fields = [
                'game_id', 'official_date', 'home_team_id', 
                'away_team_id', 'home_team_name', 'away_team_name'
            ]
            
            if not all(row_dict.get(field) for field in required_fields):
                logging.warning(f"Skipping record with missing required information: {row_dict.get('game_id', 'Unknown Game')}")
                continue
            
            try:
                if row_dict.get('home_score') is not None:
                    row_dict['home_score'] = int(row_dict['home_score'])
                if row_dict.get('away_score') is not None:
                    row_dict['away_score'] = int(row_dict['away_score'])
            except (TypeError, ValueError) as e:
                logging.warning(f"Invalid score data for game {row_dict['game_id']}: {e}")
                continue
            
            if row_dict.get('team_margin') is not None:
                try:
                    row_dict['team_margin'] = int(row_dict['team_margin'])
                except (TypeError, ValueError) as e:
                    logging.warning(f"Invalid margin data for game {row_dict['game_id']}: {e}")
                    continue
            
            results.append(row_dict)
        
        if not results:
            logging.warning(f"Query returned no results for team {team_name}")
            return []
        
        return results
    
    except exceptions.NotFound as e:
        logging.error(f"Table or dataset not found for {team_name}: {e}")
        raise
    except exceptions.BadRequest as e:
        logging.error(f"Invalid query or bad request: {e}")
        raise
    except exceptions.Forbidden as e:
        logging.error(f"Permission denied: {e}")
        raise
    except exceptions.GoogleAPIError as e:
        logging.error(f"BigQuery API error: {e}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error in fetch_team_games: {e}")
        raise

def fetch_team_player_stats(team_name: str, limit: int = 100) -> list:
    """
    Fetches player statistics for any team's games.

    Args:
        team_name (str): Team name from TEAMS dictionary
        limit (int, optional): Maximum number of records to return. Defaults to 100.

    Returns:
        list: Player statistics including standard batting metrics
    """
    
    team_id = TEAMS[team_name]
    table_name = _get_table_name(team_name)
    
    try:
        query = f"""
        SELECT
            ps.player_id,
            r.full_name,
            g.official_date as game_date,
            ps.at_bats,
            ps.hits,
            ps.home_runs,
            ps.rbi,
            ps.walks,
            ps.strikeouts,
            ps.batting_average,
            ps.on_base_percentage,
            ps.slugging_percentage
        FROM
            {table_name}.player_stats AS ps
        JOIN 
            {table_name}.roster AS r
            ON ps.player_id = r.player_id
        INNER JOIN 
            {table_name}.games AS g 
            ON ps.game_id = g.game_id
        WHERE
            (g.home_team_id = {team_id} OR g.away_team_id = {team_id})
        ORDER BY 
            g.official_date DESC
        LIMIT @limit
        """

        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("limit", "INT64", limit)
            ]
        )
        
        bq_client = bigquery.Client()
        query_job = bq_client.query(query, job_config=job_config)
        results = list(query_job.result())
        return [dict(row) for row in results]

    except Exception as e:
        logging.error(f"Error in fetch_team_player_stats for {team_name}: {e}")
        return []

def fetch_team_player_stats_by_opponent(team_name: str, opponent_team: str, limit: int = 100) -> list:
    """
    Fetches player statistics for any team's games against a specific opponent.

    Args:
        team_name (str): Team name from TEAMS dictionary
        opponent_team (str): Opponent team name
        limit (int, optional): Maximum records to return. Defaults to 100.

    Returns:
        list: Player statistics against specified opponent
    """
    
    team_id = TEAMS[team_name]
    table_name = _get_table_name(team_name)
    
    try:
        query = f"""
        SELECT
            ps.player_id,
            r.full_name,
            ps.at_bats,
            ps.hits,
            ps.home_runs,
            ps.rbi,
            ps.walks,
            ps.strikeouts,
            ps.batting_average,
            ps.on_base_percentage,
            ps.slugging_percentage
        FROM
            {table_name}.player_stats AS ps
        JOIN 
            {table_name}.roster AS r
            ON ps.player_id = r.player_id
        INNER JOIN 
            {table_name}.games AS g 
            ON ps.game_id = g.game_id
        WHERE
            (g.home_team_id = {team_id} OR g.away_team_id = {team_id})
            AND ((g.home_team_name = @opponent_team) OR (g.away_team_name = @opponent_team))
        ORDER BY 
            g.official_date DESC
        LIMIT @limit
        """

        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("opponent_team", "STRING", opponent_team),
                bigquery.ScalarQueryParameter("limit", "INT64", limit)
            ]
        )
        
        bq_client = bigquery.Client()
        query_job = bq_client.query(query, job_config=job_config)
        results = list(query_job.result())
        return [dict(row) for row in results]

    except Exception as e:
        logging.error(f"Error in fetch_team_player_stats_by_opponent for {team_name}: {e}")
        return []



def fetch_team_player_stats_by_game_type(team_name: str, game_type: str, limit: int = 100) -> list:
    """
    Fetches player statistics for any team by game type.

    Args:
        team_name (str): Team name from TEAMS dictionary
        game_type (str): Game type (R, P, etc.)
        limit (int): Max records to return. Default 100.
    """
    team_id = TEAMS[team_name]
    table_name = _get_table_name(team_name)
    
    try:
        query = f"""
        SELECT
            ps.player_id,
            r.full_name,
            ps.at_bats,
            ps.hits,
            ps.home_runs,
            ps.rbi,
            ps.walks,
            ps.strikeouts,
            ps.batting_average,
            ps.on_base_percentage,
            ps.slugging_percentage
        FROM
            {table_name}.player_stats AS ps
        JOIN 
            {table_name}.roster AS r
            ON ps.player_id = r.player_id
        INNER JOIN 
            {table_name}.games AS g 
            ON ps.game_id = g.game_id
        WHERE
            (g.home_team_id = {team_id} OR g.away_team_id = {team_id})
            AND g.game_type = @game_type
        ORDER BY 
            g.official_date DESC
        LIMIT @limit
        """

        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("game_type", "STRING", game_type),
                bigquery.ScalarQueryParameter("limit", "INT64", limit)
            ]
        )
        
        bq_client = bigquery.Client()
        query_job = bq_client.query(query, job_config=job_config)
        return [dict(row) for row in list(query_job.result())]

    except Exception as e:
        logging.error(f"Error in fetch_team_player_stats_by_game_type for {team_name}: {e}")
        return []

def fetch_team_plays(team_name: str, limit: int = 100) -> list:
    """
    Fetches plays from any team's games.

    Args:
        team_name (str): Team name from TEAMS dictionary
        limit (int): Max plays to return. Default 100.

    Returns:
        list: Play details including events, players, and timing
    """

    
    team_id = TEAMS[team_name]
    table_name = _get_table_name(team_name)
    
    try:
        query = f"""
        SELECT
            p.play_id,
            p.inning,
            p.half_inning,
            p.event,
            p.event_type,
            p.description,
            p.rbi,
            p.is_scoring_play,
            r_batter.full_name as batter_name,
            r_pitcher.full_name as pitcher_name,
            p.start_time,
            p.end_time
        FROM
            {table_name}.plays AS p
        LEFT JOIN 
            {table_name}.roster as r_batter 
            ON p.batter_id = r_batter.player_id
        LEFT JOIN 
            {table_name}.roster as r_pitcher 
            ON p.pitcher_id = r_pitcher.player_id
        INNER JOIN 
            {table_name}.games AS g 
            ON p.game_id = g.game_id
        WHERE
            g.home_team_id = {team_id} OR g.away_team_id = {team_id}
        ORDER BY 
            g.official_date DESC, p.start_time ASC
        LIMIT @limit
        """

        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("limit", "INT64", limit)
            ]
        )
        
        bq_client = bigquery.Client()
        query_job = bq_client.query(query, job_config=job_config)
        return [dict(row) for row in list(query_job.result())]

    except Exception as e:
        logging.error(f"Error in fetch_team_plays for {team_name}: {e}")
        return []
def fetch_team_plays_by_opponent(team_name: str, opponent_team: str, limit: int = 100) -> list:
    """
    Fetches plays from any team's games against specific opponent.

    Args:
        team_name (str): Team name from TEAMS dictionary
        opponent_team (str): Opponent team name
        limit (int): Max plays to return. Default 100.
    """
    team_id = TEAMS[team_name]
    table_name = _get_table_name(team_name)
    
    try:
        query = f"""
        SELECT
            p.play_id,
            p.inning,
            p.half_inning,
            p.event,
            p.event_type,
            p.description,
            p.rbi,
            p.is_scoring_play,
            r_batter.full_name as batter_name,
            r_pitcher.full_name as pitcher_name,
            p.start_time,
            p.end_time
        FROM
            {table_name}.plays AS p
        LEFT JOIN 
            {table_name}.roster as r_batter 
            ON p.batter_id = r_batter.player_id
        LEFT JOIN 
            {table_name}.roster as r_pitcher 
            ON p.pitcher_id = r_pitcher.player_id
        INNER JOIN 
            {table_name}.games AS g 
            ON p.game_id = g.game_id
        WHERE
            (g.home_team_id = {team_id} OR g.away_team_id = {team_id})
            AND ((g.home_team_name = @opponent_team) OR (g.away_team_name = @opponent_team))
        ORDER BY 
            g.official_date DESC, p.start_time ASC
        LIMIT @limit
        """

        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("opponent_team", "STRING", opponent_team),
                bigquery.ScalarQueryParameter("limit", "INT64", limit)
            ]
        )
        
        bq_client = bigquery.Client()
        query_job = bq_client.query(query, job_config=job_config)
        return [dict(row) for row in list(query_job.result())]

    except Exception as e:
        logging.error(f"Error in fetch_team_plays_by_opponent for {team_name}: {e}")
        return []

def fetch_team_plays_by_game_type(team_name: str, game_type: str, limit: int = 100) -> list:
    """
    Fetches plays from any team's games by game type.

    Args:
        team_name (str): Team name from TEAMS dictionary
        game_type (str): Game type (R, P, etc.)
        limit (int): Max plays to return. Default 100.
    """

    team_id = TEAMS[team_name]
    table_name = _get_table_name(team_name)
    
    try:
        query = f"""
        SELECT
            p.play_id,
            p.inning,
            p.half_inning,
            p.event,
            p.event_type,
            p.description,
            p.rbi,
            p.is_scoring_play,
            r_batter.full_name as batter_name,
            r_pitcher.full_name as pitcher_name,
            p.start_time,
            p.end_time
        FROM
            {table_name}.plays AS p
        LEFT JOIN 
            {table_name}.roster as r_batter 
            ON p.batter_id = r_batter.player_id
        LEFT JOIN 
            {table_name}.roster as r_pitcher 
            ON p.pitcher_id = r_pitcher.player_id
        INNER JOIN 
            {table_name}.games AS g 
            ON p.game_id = g.game_id
        WHERE
            (g.home_team_id = {team_id} OR g.away_team_id = {team_id})
            AND g.game_type = @game_type
        ORDER BY 
            g.official_date DESC, p.start_time ASC
        LIMIT @limit
        """

        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("game_type", "STRING", game_type),
                bigquery.ScalarQueryParameter("limit", "INT64", limit)
            ]
        )
        
        bq_client = bigquery.Client()
        query_job = bq_client.query(query, job_config=job_config)
        return [dict(row) for row in list(query_job.result())]

    except Exception as e:
        logging.error(f"Error in fetch_team_plays_by_game_type for {team_name}: {e}")
        return []

def fetch_team_games_by_opponent(team_name: str, opponent_team: str = 'New York Yankees', limit: int = 2) -> list:
    """
    Fetches any team's games against specific opponent.
    
    Args:
        team_name (str): Team name from TEAMS dictionary
        opponent_team (str): Opponent team name
        limit (int): Max games to return. Default 2
    """

    if not isinstance(limit, int) or limit <= 0:
        raise ValueError("limit must be a positive integer")
    if not isinstance(opponent_team, str) or not opponent_team.strip():
        raise ValueError("opponent_team must be a non-empty string")

    team_id = TEAMS[team_name]
    table_name = _get_table_name(team_name)
    
    try:
        query = f"""
        SELECT
            game_id,
            official_date,
            home_team_id,
            home_team_name,
            away_team_id,
            away_team_name,
            home_score,
            away_score,
            venue_name,
            status,
            {team_name}_win as team_win,
            {team_name}_margin as team_margin
        FROM
            {table_name}.games
        WHERE
            (home_team_id = {team_id} OR away_team_id = {team_id})
            AND ((home_team_name = @opponent_team) OR (away_team_name = @opponent_team))
        ORDER BY official_date DESC
        LIMIT @limit
        """

        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("limit", "INT64", limit),
                bigquery.ScalarQueryParameter("opponent_team", "STRING", opponent_team)
            ]
        )
        
        bq_client = bigquery.Client()
        query_job = bq_client.query(query, job_config=job_config)
        return process_game_results(query_job)

    except (exceptions.BadRequest, exceptions.Forbidden, exceptions.GoogleAPIError) as e:
        logging.error(f"API error in fetch_team_games_by_opponent for {team_name}: {e}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error in fetch_team_games_by_opponent for {team_name}: {e}")
        raise

def fetch_team_games_by_type(team_name: str, game_type: str = 'R', limit: int = 2) -> list:
    """
    Fetches any team's games by game type.
    
    Args:
        team_name (str): Team name from TEAMS dictionary
        game_type (str): Game type (R=Regular, L=League Championship, etc.)
        limit (int): Max games to return. Default 2
    """
    if not isinstance(limit, int) or limit <= 0:
        raise ValueError("limit must be a positive integer")
    if not isinstance(game_type, str) or not game_type.strip():
        raise ValueError("game_type must be a non-empty string")

    team_id = TEAMS[team_name]
    table_name = _get_table_name(team_name)
    
    try:
        query = f"""
        SELECT
            game_id,
            official_date,
            home_team_id,
            home_team_name,
            away_team_id,
            away_team_name,
            home_score,
            away_score,
            venue_name,
            status,
            {team_name}_win as team_win,
            {team_name}_margin as team_margin
        FROM
            {table_name}.games
        WHERE
            (home_team_id = {team_id} OR away_team_id = {team_id})
            AND game_type = @game_type
        ORDER BY official_date DESC
        LIMIT @limit
        """

        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("limit", "INT64", limit),
                bigquery.ScalarQueryParameter("game_type", "STRING", game_type)
            ]
        )
        
        bq_client = bigquery.Client()
        query_job = bq_client.query(query, job_config=job_config)
        return process_game_results(query_job)

    except (exceptions.BadRequest, exceptions.Forbidden, exceptions.GoogleAPIError) as e:
        logging.error(f"API error in fetch_team_games_by_type for {team_name}: {e}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error in fetch_team_games_by_type for {team_name}: {e}")
        raise

def process_game_results(query_job):
    """
    Helper function to process query results and validate data.

    Args:
        query_job: BigQuery query job result

    Returns:
        list: Processed and validated results
    """
    results = []
    for row in query_job:
        row_dict = dict(row)
        
        # Convert date object to string if present
        if row_dict.get('official_date'):
            row_dict['official_date'] = row_dict['official_date'].strftime('%Y-%m-%d')
        
        # Validate required fields
        required_fields = ['game_id', 'official_date']
        if not all(row_dict.get(field) for field in required_fields):
            logging.warning(f"Skipping record with missing required information: {row_dict.get('game_id', 'Unknown Game')}")
            continue
        
        # Validate numeric fields
        numeric_fields = ['home_score', 'away_score', 'dodgers_margin']
        for field in numeric_fields:
            try:
                if row_dict.get(field) is not None:
                    row_dict[field] = int(row_dict[field])
            except (TypeError, ValueError) as e:
                logging.warning(f"Invalid {field} data for game {row_dict['game_id']}: {e}")
                row_dict[field] = None
        
        results.append(row_dict)
    
    if not results:
        logging.warning("Query returned no results")
        return []
    
    return results
    


def fetch_player_game_stats(team_name: str, game_ids: list = None, player_ids: list = None) -> list:
    """
    Fetches player statistics for any team's games/players.

    Args:
        team_name (str): Team name from TEAMS dictionary
        game_ids (list, optional): Game IDs to filter by
        player_ids (list, optional): Player IDs to filter by
    """
    if game_ids is not None and not isinstance(game_ids, list):
        raise ValueError("game_ids must be a list or None")
    if player_ids is not None and not isinstance(player_ids, list):
        raise ValueError("player_ids must be a list or None")

    table_name = _get_table_name(team_name)
    
    try:
        query = f"""
        SELECT
            ps.player_id,
            r.full_name,
            ps.at_bats,
            ps.hits,
            ps.home_runs,
            ps.rbi,
            ps.walks,
            ps.strikeouts,
            ps.batting_average,
            ps.on_base_percentage,
            ps.slugging_percentage
        FROM
            {table_name}.player_stats AS ps
        JOIN 
            {table_name}.roster AS r
        ON 
            ps.player_id = r.player_id
        """

        query_params = []
        where_conditions = []

        if game_ids:
            where_conditions.append("ps.game_id IN UNNEST(@game_id_list)")
            query_params.append(
                bigquery.ArrayQueryParameter("game_id_list", "STRING", game_ids)
            )

        if player_ids:
            where_conditions.append("ps.player_id IN UNNEST(@player_id_list)")
            query_params.append(
                bigquery.ArrayQueryParameter("player_id_list", "STRING", player_ids)
            )

        if where_conditions:
            query += "\nWHERE " + " AND ".join(where_conditions)

        job_config = bigquery.QueryJobConfig(query_parameters=query_params)
        bq_client = bigquery.Client()
        query_job = bq_client.query(query, job_config=job_config)
        
        results = []
        for row in query_job:
            row_dict = dict(row)
            
            if not all(row_dict.get(field) for field in ['player_id', 'full_name']):
                logging.warning(f"Skipping record with missing required information: {row_dict.get('player_id', 'Unknown Player')}")
                continue
            
            numeric_fields = [
                'at_bats', 'hits', 'home_runs', 'rbi', 'walks', 'strikeouts',
                'batting_average', 'on_base_percentage', 'slugging_percentage'
            ]
            
            for field in numeric_fields:
                try:
                    if row_dict.get(field) is not None:
                        if field in ['batting_average', 'on_base_percentage', 'slugging_percentage']:
                            row_dict[field] = float(row_dict[field])
                        else:
                            row_dict[field] = int(row_dict[field])
                except (TypeError, ValueError) as e:
                    logging.warning(f"Invalid {field} data for player {row_dict['full_name']}: {e}")
                    row_dict[field] = None
            
            results.append(row_dict)
        
        if not results:
            logging.warning(f"Query returned no results for {team_name}")
            return []
        
        return results
    
    except (exceptions.NotFound, exceptions.BadRequest, exceptions.Forbidden, exceptions.GoogleAPIError) as e:
        logging.error(f"API error in fetch_player_game_stats for {team_name}: {e}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error in fetch_player_game_stats for {team_name}: {e}")
        raise

def fetch_player_plays(player_name: str, team_name: str, limit: int = 100) -> list:
    """
    Fetches play-by-play data for a specific player from any team's games.
    
    Args:
        player_name (str): Full name of the player
        team_name (str): Team name from TEAMS dictionary
        limit (int, optional): Maximum number of plays to return. Defaults to 100.
        
    Returns:
        list: List of dictionaries containing play-by-play data
    """
    
    team_id = TEAMS[team_name]
    table_name = _get_table_name(team_name)
    
    try:
        query = f"""
        SELECT
            p.play_id,
            p.inning,
            p.half_inning,
            p.event,
            p.event_type,
            p.description,
            p.start_time,
            g.official_date as game_date
        FROM
            {table_name}.plays AS p
        INNER JOIN 
            {table_name}.games AS g 
            ON p.game_id = g.game_id
        INNER JOIN
            {table_name}.roster AS r
            ON (p.batter_id = r.player_id OR p.pitcher_id = r.player_id)
        WHERE
            r.full_name = @player_name
            AND (g.home_team_id = {team_id} OR g.away_team_id = {team_id})
        ORDER BY 
            g.official_date DESC,
            p.start_time ASC
        LIMIT @limit
        """

        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("player_name", "STRING", player_name),
                bigquery.ScalarQueryParameter("limit", "INT64", limit)
            ]
        )
        
        bq_client = bigquery.Client()
        query_job = bq_client.query(query, job_config=job_config)
        results = list(query_job.result())

        formatted_results = []
        for row in results:
            row_dict = dict(row)
            if 'start_time' in row_dict and row_dict['start_time']:
                row_dict['start_time'] = row_dict['start_time'].isoformat()
            if 'game_date' in row_dict and row_dict['game_date']:
                row_dict['game_date'] = row_dict['game_date'].isoformat()
            formatted_results.append(row_dict)
        return formatted_results

    except Exception as e:
        logging.error(f"Error in fetch_player_plays_generic: {e}")
        return []

def fetch_player_plays_by_opponent(player_name: str, team_name: str, opponent_team: str, limit: int = 100) -> list:
    """
    Fetches play-by-play data for a specific player against a specific opponent.
    
    Args:
        player_name (str): Full name of the player
        team_name (str): Team name from TEAMS dictionary
        opponent_team (str): Name of the opponent team
        limit (int, optional): Maximum number of plays to return. Defaults to 100.
    """

    team_id = TEAMS[team_name]
    table_name = _get_table_name(team_name)
    
    try:
        query = f"""
        SELECT
            p.play_id,
            p.inning,
            p.half_inning,
            p.event,
            p.event_type,
            p.description,
            p.start_time,
            g.official_date as game_date
        FROM
            {table_name}.plays AS p
        INNER JOIN 
            {table_name}.games AS g 
            ON p.game_id = g.game_id
        INNER JOIN
            {table_name}.roster AS r
            ON p.batter_id = r.player_id
        WHERE
            r.full_name = @player_name
            AND (g.home_team_id = {team_id} OR g.away_team_id = {team_id})
            AND ((g.home_team_name = @opponent_team) OR (g.away_team_name = @opponent_team))
        ORDER BY 
            g.official_date DESC,
            p.start_time ASC
        LIMIT @limit
        """

        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("player_name", "STRING", player_name),
                bigquery.ScalarQueryParameter("opponent_team", "STRING", opponent_team),
                bigquery.ScalarQueryParameter("limit", "INT64", limit)
            ]
        )
        
        bq_client = bigquery.Client()
        query_job = bq_client.query(query, job_config=job_config)
        results = list(query_job.result())
        return [dict(row) for row in results]

    except Exception as e:
        logging.error(f"Error in fetch_player_plays_by_opponent_generic: {e}")
        return []

def fetch_player_plays_by_game_type(player_name: str, team_name: str, game_type: str, limit: int = 100) -> list:
    """
    Fetches play-by-play data for a specific player from games of a specific type.
    
    Args:
        player_name (str): Full name of the player
        team_name (str): Team name from TEAMS dictionary
        game_type (str): Type of game (R for Regular Season, P for Postseason, etc.)
        limit (int, optional): Maximum number of plays to return. Defaults to 100.
    """

    team_id = TEAMS[team_name]
    table_name = _get_table_name(team_name)
    
    try:
        query = f"""
        SELECT
            p.play_id,
            p.inning,
            p.half_inning,
            p.event,
            p.event_type,
            p.description,
            p.start_time,
            g.official_date as game_date
        FROM
            {table_name}.plays AS p
        INNER JOIN 
            {table_name}.games AS g 
            ON p.game_id = g.game_id
        INNER JOIN
            {table_name}.roster AS r
            ON p.batter_id = r.player_id
        WHERE
            r.full_name = @player_name
            AND (g.home_team_id = {team_id} OR g.away_team_id = {team_id})
            AND g.game_type = @game_type
        ORDER BY 
            g.official_date DESC,
            p.start_time ASC
        LIMIT @limit
        """

        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("player_name", "STRING", player_name),
                bigquery.ScalarQueryParameter("game_type", "STRING", game_type),
                bigquery.ScalarQueryParameter("limit", "INT64", limit)
            ]
        )
        
        bq_client = bigquery.Client()
        query_job = bq_client.query(query, job_config=job_config)
        results = list(query_job.result())
        return [dict(row) for row in results]

    except Exception as e:
        logging.error(f"Error in fetch_player_plays_by_game_type_generic: {e}")
        return []


def generate_mlb_podcasts(contents: str) -> dict:
   
    client = genai.Client(vertexai=True, project="gem-rush-007", location="us-central1")
    MODEL_ID = "gemini-2.0-flash-exp"

    # Structure the prompt to explicitly request tool usage
    structured_prompt = f"""
         You are a sophisticated sports podcast script generator. Your task is to create compelling scripts based on user preferences and data. Here is the breakdown of your responsibilities:

    **Step 1: Understand User Preferences**
        *   Carefully analyze the user's request provided in the "Question" field. This will include information about the team, player, the time frame, type of game (if specified), and any specific aspects they want to be covered.
        *   Identify what type of data is most important to the user (e.g. player highlights, team game summaries, game analysis).
        * Determine which information to highlight based on the data provided and user preferences.

    **Step 2: Data Fetching and Analysis**
        *   Based on your understanding of the user preferences, select the most appropriate tools from the available list to fetch the necessary data. Use multiple tools if necessary to gather all the information.
        *   Analyze the fetched data, focus on identifying key events, stats, and interesting information for the podcast script (e.g., home runs, close plays, wins, losses, player performance).
        *   If user select games from the past, make sure to use this as the primary source of information.
        *  If the user specifies a particular player, then prioritize their performance in the game.

    **Step 3: Multi-Speaker Script Generation**
        *   Create a script with multiple speakers. At a minimum you must use the following three speakers.
            *   **Play-by-play Announcer:** This speaker describes the events as they happen. Use a neutral and clear voice.
            *   **Color Commentator:** This speaker provides analysis and insights. Use a voice that is more insightful and analytical.
            *   **Player Quotes (Simulated):** This speaker uses simulated quotes from the players involved in a play. Use a more casual and personal tone for this speaker.
        *  Structure the script so that for each key event you utilize all three speakers to convey the information.
        *    Include the following information in your script for each play, if available.
            *   Player names
            *   Team names
            *   Inning
            *   Description of what happened
            *    Other relevant stats if available
        *   Use transitions between plays to make it sound like a coherent narrative.
        *  Keep a neutral tone and try to avoid personal opinion unless specified by the user.

    **Step 4: Language Support**
        *  Translate the final script and any associated text using the provided translation tools to support the users preferred language.
        *   The language may or may not be specified by the user. If the language is not specified assume the user speaks English.
        *   All components of the script should be translated including any text based data you will use to generate the podcast.

    
    **Step 5: Audio Generation Output**
        *   Format the final output so that it contains the speaker, and the content.
        *   Your response must be a valid JSON array without any markdown formatting or code blocks.
        *   For example:
            [
                {{
                    "speaker": "Play-by-play Announcer",
                    "text": "Here's the pitch, swung on and a long drive..."
                }},
                {{
                    "speaker": "Color Commentator",
                    "text": "Unbelievable power from [Player Name] there, that was a no doubter."
                }},
                {{
                    "speaker": "Player Quotes",
                    "text": "I knew I was gonna hit that out of the park!"
                }}
            ]
        *    Provide all the output in a single json array without any markdown formatting.
        
    **Your Output must be a pure JSON array without any markdown code blocks or formatting. Just the raw JSON.**

    Question: {contents}

    Remember to prioritize all the steps, and ensure you generate a compelling and informative podcast script.
    """

    try:
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=structured_prompt,
            config=GenerateContentConfig(
                tools=[

                    fetch_player_plays,
                    fetch_team_games,
                    fetch_team_games_by_opponent,
                    fetch_team_games_by_type,
                    fetch_team_plays,
                    fetch_team_plays_by_opponent,
                    fetch_team_plays_by_game_type,
                    fetch_team_player_stats,
                    fetch_team_player_stats_by_game_type,
                    fetch_team_player_stats_by_opponent,
                    fetch_player_plays_by_game_type,
                    fetch_player_plays_by_opponent,
                ],
                temperature=0,
            ),
        )


        try:
            # Clean the response text by removing markdown code block syntax
            text = response.text
            if text.startswith("```"):
                # Find the first newline after the opening ```
                start_idx = text.find("\n") + 1
                # Find the last ``` and exclude everything after it
                end_idx = text.rfind("```")
                if end_idx == -1:  # If no closing ```, just remove the opening
                    text = text[start_idx:]
                else:
                    text = text[start_idx:end_idx].strip()
            
            # Remove any "json" or other language identifier that might appear
            text = text.replace("json\n", "")
            
            # Parse the cleaned JSON
            text_response = json.loads(text)
            return text_response
        except json.JSONDecodeError as e:
            logging.error(f"JSON Decode Error in generate_mlb_analysis: {e}, response was {text}")
            return {
                "error": f"JSON Decode Error in generate_mlb_analysis: {e}, please check the logs"
            }
    except Exception as e:
        logging.error(f"Error in generate_mlb_analysis: {e}")
        return {
            "error": f"An error occurred: {e}",
        }