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

safety_settings = [
    SafetySetting(
        category="HARM_CATEGORY_DANGEROUS_CONTENT",
        threshold="BLOCK_LOW_AND_ABOVE",
    ),
    SafetySetting(
        category="HARM_CATEGORY_HARASSMENT",
        threshold="BLOCK_LOW_AND_ABOVE",
    ),
    SafetySetting(
        category="HARM_CATEGORY_HATE_SPEECH",
        threshold="BLOCK_LOW_AND_ABOVE",
    ),
    SafetySetting(
        category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
        threshold="BLOCK_LOW_AND_ABOVE",
    ),
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


def fetch_team_games(team_name: str, limit: int = 2) -> list:
    """
    Fetches the most recent games (limited by 'limit') for a specified team 
    using plays data to determine game recency.

    Args:
        team_name (str): The team name as it appears in the TEAMS dictionary
        limit (int, optional): The maximum number of games to return. Defaults to 2.

    Returns:
        list: A list of dictionaries containing game details
    """
    
    team_id = TEAMS[team_name]
    table_name = _get_table_name(team_name)

    try:
      query = f"""
          SELECT
              g.game_id,
              g.official_date,
              g.home_team_id,
              g.home_team_name,
              g.away_team_id,
              g.away_team_name,
              g.home_score,
              g.away_score,
              g.venue_name,
              g.status,
              {team_name}_win as team_win,
              {team_name}_margin as team_margin
          FROM
              {table_name}.games AS g
          INNER JOIN
              (SELECT
                game_id,
                MAX(end_time) AS max_end_time
              FROM
                  {table_name}.plays
              GROUP BY game_id
              ORDER BY max_end_time DESC
              LIMIT @limit
            ) AS subquery
              ON g.game_id = subquery.game_id
          WHERE
            (g.home_team_id = {team_id} OR g.away_team_id = {team_id})
          ORDER BY subquery.max_end_time DESC
      """
    
      job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("limit", "INT64", limit),
            ]
        )
        
      bq_client = bigquery.Client()
      query_job = bq_client.query(query, job_config=job_config)
      return process_game_results(query_job)
  
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
    Fetches player statistics for the most recent games of a team, ordered by play end time.

    Args:
        team_name (str): Team name from TEAMS dictionary
        limit (int, optional): Maximum number of records to return. Defaults to 100.

    Returns:
        list: A list of dictionaries containing player stats.
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
            INNER JOIN (
              SELECT 
                  game_id,
                  MAX(end_time) as max_end_time
              FROM
                 {table_name}.plays
              GROUP BY game_id
              ORDER BY max_end_time DESC
             ) AS subquery
             ON g.game_id = subquery.game_id
            WHERE
                (g.home_team_id = {team_id} OR g.away_team_id = {team_id})
           ORDER BY subquery.max_end_time DESC
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
        list: A list of dictionaries, each containing the following keys:
            - player_id: Unique identifier for the player.
            - full_name: The player's full name.
            - game_date: The official date of the game.
            - at_bats: Number of at-bats.
            - hits: Number of hits.
            - home_runs: Number of home runs.
            - rbi: Runs batted in.
            - walks: Number of walks.
            - strikeouts: Number of strikeouts.
            - batting_average: Batting average.
            - on_base_percentage: On-base percentage.
            - slugging_percentage: Slugging percentage.
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
            INNER JOIN (
              SELECT 
                  game_id,
                  MAX(end_time) as max_end_time
              FROM
                 {table_name}.plays
              GROUP BY game_id
              ORDER BY max_end_time DESC
             ) AS subquery
             ON g.game_id = subquery.game_id
            WHERE
                (g.home_team_id = {team_id} OR g.away_team_id = {team_id})
                AND ((g.home_team_name = @opponent_team) OR (g.away_team_name = @opponent_team))
           ORDER BY subquery.max_end_time DESC
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

    Returns:
        list: A list of dictionaries, each containing the following keys:
            - player_id: Unique identifier for the player.
            - full_name: The player's full name.
            - game_date: The official date of the game.
            - at_bats: Number of at-bats.
            - hits: Number of hits.
            - home_runs: Number of home runs.
            - rbi: Runs batted in.
            - walks: Number of walks.
            - strikeouts: Number of strikeouts.
            - batting_average: Batting average.
            - on_base_percentage: On-base percentage.
            - slugging_percentage: Slugging percentage.
    """
    team_id = TEAMS[team_name]
    table_name = _get_table_name(team_name)
    
    try:
        query = f"""
        SELECT
            ps.player_id,
            g.official_date as game_date,
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
            INNER JOIN (
              SELECT 
                  game_id,
                  MAX(end_time) as max_end_time
              FROM
                 {table_name}.plays
              GROUP BY game_id
              ORDER BY max_end_time DESC
             ) AS subquery
             ON g.game_id = subquery.game_id
            WHERE
                (g.home_team_id = {team_id} OR g.away_team_id = {team_id})
                AND g.game_type = @game_type
           ORDER BY subquery.max_end_time DESC
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
        list: A list of dictionaries, each containing the following keys:
            - play_id: Unique identifier for the play.
            - inning: The inning in which the play occurred.
            - half_inning: Indicates whether the play occurred in the top or bottom of the inning.
            - event: The event that occurred (e.g., a hit, a strikeout).
            - event_type: The type or classification of the event.
            - description: A textual description of the play.
            - rbi: The number of runs batted in as a result of the play.
            - is_scoring_play: Boolean indicating if the play resulted in a score.
            - batter_name: Full name of the batter involved in the play.
            - pitcher_name: Full name of the pitcher involved in the play.
            - start_time: The start time of the play.
            - end_time: The end time of the play.
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
            p.end_time DESC
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

    Returns:
        list: A list of dictionaries, each containing the following keys:
            - play_id: Unique identifier for the play.
            - inning: The inning in which the play occurred.
            - half_inning: Indicates whether the play occurred in the top or bottom of the inning.
            - event: The event that occurred (e.g., a hit, a strikeout).
            - event_type: The type or classification of the event.
            - description: A textual description of the play.
            - rbi: The number of runs batted in as a result of the play.
            - is_scoring_play: Boolean indicating if the play resulted in a score.
            - batter_name: Full name of the batter involved in the play.
            - pitcher_name: Full name of the pitcher involved in the play.
            - start_time: The start time of the play.
            - end_time: The end time of the play.
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
            p.end_time DESC, p.start_time ASC
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

    Returns:
        list: A list of dictionaries, each containing the following keys:
            - play_id: Unique identifier for the play.
            - inning: The inning in which the play occurred.
            - half_inning: Indicates whether the play occurred in the top or bottom of the inning.
            - event: The event that occurred (e.g., a hit, a strikeout).
            - event_type: The type or classification of the event.
            - description: A textual description of the play.
            - rbi: The number of runs batted in as a result of the play.
            - is_scoring_play: Boolean indicating if the play resulted in a score.
            - batter_name: Full name of the batter involved in the play.
            - pitcher_name: Full name of the pitcher involved in the play.
            - start_time: The start time of the play.
            - end_time: The end time of the play.
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
             p.end_time DESC, p.start_time ASC
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

    Returns:
        list: A list of dictionaries, each containing the following keys:
            - game_id: Unique identifier for the game.
            - official_date: The official date of the game.
            - home_team_id: The ID of the home team.
            - home_team_name: The name of the home team.
            - away_team_id: The ID of the away team.
            - away_team_name: The name of the away team.
            - home_score: The final score for the home team.
            - away_score: The final score for the away team.
            - venue_name: The name of the venue where the game was played.
            - status: The status of the game (e.g., completed, postponed).
            - team_win: Boolean or indicator whether the specified team won the game.
            - team_margin: The score margin by which the specified team won or lost the game.
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
              {table_name}.games AS g
        INNER JOIN
              (SELECT
                game_id,
                MAX(end_time) AS max_end_time
              FROM
                  {table_name}.plays
              GROUP BY game_id
              ORDER BY max_end_time DESC
              LIMIT @limit
            ) AS subquery
              ON g.game_id = subquery.game_id
        WHERE
            (g.home_team_id = {team_id} OR g.away_team_id = {team_id})
            AND ((home_team_name = @opponent_team) OR (away_team_name = @opponent_team))
        ORDER BY subquery.max_end_time DESC
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

    Returns:
        list: A list of dictionaries, each containing the following keys:
            - game_id: Unique identifier for the game.
            - official_date: The official date of the game.
            - home_team_id: The ID of the home team.
            - home_team_name: The name of the home team.
            - away_team_id: The ID of the away team.
            - away_team_name: The name of the away team.
            - home_score: The final score for the home team.
            - away_score: The final score for the away team.
            - venue_name: The name of the venue where the game was played.
            - status: The status of the game (e.g., completed, postponed).
            - team_win: Boolean or indicator whether the specified team won the game.
            - team_margin: The score margin by which the specified team won or lost the game.
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
              {table_name}.games AS g
        INNER JOIN
              (SELECT
                game_id,
                MAX(end_time) AS max_end_time
              FROM
                  {table_name}.plays
              GROUP BY game_id
              ORDER BY max_end_time DESC
              LIMIT @limit
            ) AS subquery
              ON g.game_id = subquery.game_id
        WHERE
            (g.home_team_id = {team_id} OR g.away_team_id = {team_id})
            AND game_type = @game_type
        ORDER BY subquery.max_end_time DESC
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
    


def fetch_player_game_stats(team_name: str, game_ids: list = None,  limit: int = 100, player_ids: list = None) -> list:
    """
    Fetches player statistics for any team's games/players.

    Args:
        team_name (str): Team name from TEAMS dictionary
        limit (int, optional): Maximum number of records to return.
        game_ids (list, optional): Game IDs to filter by
        player_ids (list, optional): Player IDs to filter by

    Returns:
        list: A list of dictionaries, each containing the following keys:
            - player_id: Unique identifier for the player.
            - full_name: The player's full name.
            - game_date: The official date of the game.
            - at_bats: Number of at-bats.
            - hits: Number of hits.
            - home_runs: Number of home runs.
            - rbi: Runs batted in.
            - walks: Number of walks.
            - strikeouts: Number of strikeouts.
            - batting_average: Batting average.
            - on_base_percentage: On-base percentage.
            - slugging_percentage: Slugging percentage.
    """
    if game_ids is not None and not isinstance(game_ids, list):
        raise ValueError("game_ids must be a list or None")
    if player_ids is not None and not isinstance(player_ids, list):
        raise ValueError("player_ids must be a list or None")

    table_name = _get_table_name(team_name)
    team_id = TEAMS[team_name]
    
    try:
        query = f"""
        SELECT
            ps.player_id,
            g.official_date as game_date,
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
            INNER JOIN 
                {table_name}.games AS g 
                ON ps.game_id = g.game_id
            INNER JOIN (
              SELECT 
                  game_id,
                  MAX(end_time) as max_end_time
              FROM
                 {table_name}.plays
              GROUP BY game_id
              ORDER BY max_end_time DESC
             ) AS subquery
             ON g.game_id = subquery.game_id
            WHERE
                (g.home_team_id = {team_id} OR g.away_team_id = {team_id})
                AND g.game_type = @game_type
           ORDER BY subquery.max_end_time DESC
           LIMIT @limit    
        """

        query_params = [bigquery.ScalarQueryParameter("limit", "INT64", limit)]
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
        list: A list of dictionaries, each containing the following keys:
            - play_id: Unique identifier for the play.
            - inning: The inning in which the play occurred.
            - half_inning: Indicates whether the play occurred in the top or bottom of the inning.
            - event: The event that occurred (e.g., a hit, a strikeout).
            - event_type: The type or classification of the event.
            - description: A textual description of the play.
            - start_time: The start time of the play.
            - game_date: The official date of the game in which the play occurred.
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
            p.end_time DESC,
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

    Returns:
        list: A list of dictionaries, each containing the following keys:
            - play_id: Unique identifier for the play.
            - inning: The inning in which the play occurred.
            - half_inning: Indicates whether the play occurred in the top or bottom of the inning.
            - event: The event that occurred (e.g., a hit, a strikeout).
            - event_type: The type or classification of the event.
            - description: A textual description of the play.
            - start_time: The start time of the play.
            - game_date: The official date of the game in which the play occurred.        
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
            p.end_time DESC,
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

    Returns:
        list: A list of dictionaries, each containing the following keys:
            - play_id: Unique identifier for the play.
            - inning: The inning in which the play occurred.
            - half_inning: Indicates whether the play occurred in the top or bottom of the inning.
            - event: The event that occurred (e.g., a hit, a strikeout).
            - event_type: The type or classification of the event.
            - description: A textual description of the play.
            - start_time: The start time of the play.
            - game_date: The official date of the game in which the play occurred.
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
            p.end_time DESC,
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


def evaluate_podcast_script(script: str, original_question: str) -> dict:
    """
    Evaluates a podcast script using Gemini, providing feedback on various aspects.

    Args:
        script (str): The JSON formatted podcast script to evaluate.
        original_question (str): The user's original prompt that was used to generate the script.

    Returns:
        dict: A dictionary containing the evaluation results.
    """
    client = genai.Client(vertexai=True, project="gem-rush-007", location="us-central1")
    MODEL_ID = "gemini-2.0-flash-exp"

    evaluator_prompt = f"""
    You are a highly skilled podcast script evaluator. Your task is to critically assess a given sports podcast script and provide detailed feedback. Consider the following aspects:

    **Evaluation Criteria:**

    1.  **Relevance to the Original Question:**
        *   How well does the script address the original user question?
        *   Does it focus on the key aspects requested, including specific teams, players, and timeframes?
        *   Does it effectively extract all relevant information from the user's question?
        *   Does the script address the question in a comprehensive manner?
        *   Is the scope of the script appropriate for the given user request?
        *   Rate the relevance of the script to the user's question on a scale from 1 to 10, where 1 is not relevant at all and 10 is extremely relevant.

    2.  **Data Accuracy and Completeness:**
        *   Is the data presented in the script accurate and supported by the available data?
        *   Are key game events, stats, and player information correctly presented?
        *   Are there any factual errors or missing pieces of crucial information?
        *  If some data was missing did the script appropriately call out that data was missing?
        *   Does the script use data to effectively enhance the script?
        * Rate the accuracy of the data on a scale of 1 to 10, where 1 indicates extremely inaccurate and 10 indicates perfectly accurate.

    3.  **Multi-Speaker Script Quality:**
        *   How effectively does the script utilize the three speaker roles (Play-by-play Announcer, Color Commentator, Simulated Player Quotes)?
        *   Is each speaker role distinct, with clear variations in language and tone?
        *   Does the script maintain a natural and engaging conversation flow between the speakers?
         * Does the script seamlessly transition between different play segments?
        *   Are simulated player quotes realistic and fitting for the context?
        *   Rate the overall quality of the multi-speaker script on a scale of 1 to 10, where 1 indicates poor use of speakers and 10 indicates excellent use of speakers.

    4.  **Script Flow and Coherence:**
        *   Does the script have a logical flow and a clear narrative structure?
        *   Are transitions between plays smooth and coherent?
        *   Does the script provide a good listening experience?
        *    Are the plays and events presented in a logical order?
        *   Rate the script's flow and coherence on a scale of 1 to 10, where 1 indicates disjointed and 10 indicates a perfectly coherent flow.

    5. **Use of Edge Case Logic:**
        * Does the script appropriately handle edge cases such as data gaps or unexpected situations?
        *   Does the script fail gracefully when faced with missing or incomplete data?
        *    Does the script accurately report any error conditions?
        *    Rate the edge case handling of the script on a scale of 1 to 10, where 1 indicates poor handling and 10 indicates excellent handling.

    **Output:**
    *   Provide a detailed evaluation report. For each criterion provide a score from 1-10, where 1 is the lowest score and 10 is the highest score. Provide a summary paragraph explaining the evaluation for each category and the rational for why you rated each category as you did.
     * Format the output as a valid JSON object.

    **Input:**
        *   Original User Question: {original_question}
        *   Podcast Script: {script}

    Respond with a valid JSON object.
    """


    try:
        google_search_tool = Tool(google_search=GoogleSearch())
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=evaluator_prompt,
            config=GenerateContentConfig(
                temperature=0.2,
                top_p=0.95,
                top_k=40,
                max_output_tokens=2048,
                tools=[google_search_tool],
                safety_settings=safety_settings,
            ),
        )
        try:
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
             logging.error(f"JSON Decode Error in evaluate_podcast_script: {e}, response was {text}")
             return {
                "error": f"JSON Decode Error in evaluate_podcast_script: {e}, please check the logs"
            }
    except Exception as e:
        logging.error(f"Error in evaluate_podcast_script: {e}")
        return {
            "error": f"An error occurred: {e}",
        }

def generate_mlb_podcasts(contents: str) -> dict:
   
    client = genai.Client(vertexai=True, project="gem-rush-007", location="us-central1")
    MODEL_ID = "gemini-2.0-pro-exp-02-05"

    structured_prompt = """
You are an expert sports podcast script generator.  Your primary task is to create scripts based on **data retrieved from the provided tools**.  Do *not* invent game data.  Rely *exclusively* on the tools.

**Overall Goal:** Create a podcast script that *accurately* reflects the data returned by the tools, addressing the user's request.

**ABSOLUTELY MANDATORY: PRE-FLIGHT CHECKLIST (Part of JSON Output)**

Before generating the podcast script, you MUST create a "pre-flight checklist" and include it as the FIRST element in the JSON output array. This checklist verifies that you understand the request and are using the correct tools.

The checklist MUST be a JSON object with the following keys:

*   `"checklist"`:  Set this to `true` to indicate it's the checklist.
*   `"question"`: The *exact* user question.
*   `"games_to_cover"`: The number of games to cover (e.g., 1, 2, etc.).
*   `"teams"`: An array of team names involved (e.g., `["Guardians", "White Sox"]`).
*   `"tool_calls"`: An *array* of objects.  Each object describes a *single* tool call:
    *   `"function"`: The *exact* name of the tool function (e.g., "fetch_team_games").
    *   `"arguments"`: A *dictionary* of arguments to pass to the function (e.g., `{"team_name": "Guardians", "limit": 2}`).

**Example Checklist (Illustrative):**

```json
[
    {{
        "checklist": true,
        "question": "Generate a podcast about the Cleveland Guardians. Cover the last 2 games played by the Cleveland Guardians. Generate the podcast script in Spanish.",
        "games_to_cover": 2,
        "teams": ["Guardians"],
        "tool_calls": [
            {{
                "function": "fetch_team_games",
                "arguments": {{"team_name": "guardians", "limit": 2}}
            }}
        ]
    }},
        {{
        "speaker": "Narrador de jugada por jugada",
        "text": "¡Bienvenidos! Hoy repasaremos los últimos dos partidos de los Cleveland Guardians. Primero, el partido del 11-05-2024 contra los Chicago White Sox. El marcador final fue 3-1, victoria para los Guardians."
    }},
    {{
        "speaker": "Comentarista de color",
        "text": "Un partido muy reñido.  Andrés Giménez conectó un doble importante, impulsando una carrera."
    }},
    {{
       "speaker": "Citas de Jugadores",
       "text": "Solo estaba tratando de hacer un buen contacto con la pelota."
    }}
]
```

**Step 2: User Request and Tool Selection (MANDATORY)**

1.  **Analyze the Question:** Understand the user's request (team, opponent, game type, date, etc.).
2.  **Identify Required Data:** Determine *exactly* what data is needed to answer the question.  Be specific (e.g., "game scores," "player stats for specific games," "plays involving a specific player").
3.  **Choose the Correct Tool(s):**  Based on the required data, select the *appropriate* tool function(s) to call.  For example:
    *   To get game information: Use `fetch_team_games`, `fetch_team_games_by_opponent`, or `fetch_team_games_by_type`.
    *   To get player statistics: Use `fetch_team_player_stats`, `fetch_team_player_stats_by_opponent`, or `fetch_team_player_stats_by_game_type`.
    *   To get play-by-play data: Use `fetch_team_plays`, `fetch_team_plays_by_opponent`, or `fetch_team_plays_by_game_type`.
4.  **DEBUGGING STEP (CRITICAL):** Before generating *any* script content, include a section in your response (as a comment or in a separate JSON field *if* you were generating JSON at this stage) that *lists*:
    *   The exact question.
    *   The data needed to answer the question.
    *   The *specific* tool function(s) you will call, along with the *exact arguments* you will pass to them (e.g., `fetch_team_games(team_name='guardians', limit=2)`).
    * This step is *essential* for debugging.

**Step 3: Data Retrieval (MANDATORY)**

1.  **Call the Tools:** Execute the selected tool function(s) with the correct arguments.
2.  **Store the Results:** Capture the *exact* output returned by the tool function(s).  Do *not* modify the results at this stage.

**Step 3: Script Generation (Using ONLY Retrieved Data)**

1.  **Speaker Roles:** Use the following speaker roles:
    *   `"Narrador de jugada por jugada"` (Play-by-play Announcer)
    *   `"Comentarista de color"` (Color Commentator)
    *   `"Citas de Jugadores"` (Simulated Player Quotes)
2.  **Structure:** Create a script that presents the information from the *tool results* in a logical and engaging way.
3.  **Data Integration:**  Incorporate the data retrieved in Step 2 *directly* into the script. Include:
    *   Game dates (using Spanish format: `dd-mm-yyyy`).
    *   Final scores.
    *   Team names.
    *   Player names.
    *   Relevant statistics (from the tool output).
    *   Descriptions of key plays (if play-by-play data is available).
4.  **Transitions:** Use phrases to connect different parts of the script smoothly.
5.  **Spanish Language:** Use lively and engaging language appropriate for Spanish sports broadcasting. Emphasize game dates and scores: "la fecha del partido" and "el marcador final".

**Step 5: JSON Output (Strict Format)**

1.  **Format:** Output a *single* JSON array.  Each element in the array *must* be an object with *exactly* two keys:
    *   `"speaker"`:  One of the speaker roles listed above.
    *   `"text"`:  The text for that speaker segment.
2.  **No Extra Content:** Do *not* include any markdown, code blocks, introductory text, or explanations outside of the JSON array.


    *   **Edge Case Management:**
        *   **JSON Errors:** If there is a problem creating the json object, then return a json object with an error message.
**Your Output must be a pure JSON array without any markdown code blocks or formatting. Just the raw JSON.**

Question: {contents}

Prioritize the correct execution of each step to ensure the creation of a high-quality, informative, and engaging podcast script, fully tailored to the user's request. Be sure to consider any edge cases in the process. *Specifically, prioritize finding and using the most recent relevant data if the initially requested data is unavailable.*
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
                top_p=0.95,
                top_k=40,
                max_output_tokens=2048,
                safety_settings=safety_settings,
            ),
        )


        try:
            # Clean the response text by removing markdown code block syntax
            text = response.text
            logging.debug(f"Raw response text: {text}") 
            print(f"Raw response text: {text}")
            if text.startswith("```"):
                # Find the first newline after the opening ```
                start_idx = text.find("\n") + 1
                # Find the last ``` and exclude everything after it
                end_idx = text.rfind("```")
                if end_idx == -1:  # If no closing ```, just remove the opening
                    text = text[start_idx:]
                else:
                    text = text[start_idx:end_idx].strip()
            logging.debug(f"Text after markdown removal: {text}")

            print(f"Text after markdown removal: {text}")
            
            # Remove any "json" or other language identifier that might appear
            text = text.replace("json\n", "")
            logging.debug(f"Text after json removal: {text}") 
            print(f"Text after json removal: {text}")
            
            # Parse the cleaned JSON
            text_response = json.loads(text)
            evaluation = evaluate_podcast_script(text, contents)
            print(evaluation)
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


