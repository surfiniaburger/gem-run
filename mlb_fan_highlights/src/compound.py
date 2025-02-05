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




def fetch_team_games(team_name: str, limit: int = 2, specific_date: Optional[str] = None) -> list:
    """
    Fetches the most recent games (limited by 'limit') for a specified team
    using plays data to determine game recency, with optional filtering by a specific date.

    Args:
        team_name (str): The team name as it appears in the TEAMS dictionary
        limit (int, optional): The maximum number of games to return. Defaults to 2.
        specific_date (Optional[str], optional): A specific date in 'YYYY-MM-DD' format to filter games.

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
        """

        # Add date filtering directly within the subquery
        if specific_date:
            query += f" WHERE DATE(start_time) = '{specific_date}' "

        query += f"""
                GROUP BY game_id
                ORDER BY max_end_time DESC
                LIMIT @limit
                ) AS subquery
                ON g.game_id = subquery.game_id
            WHERE
                (g.home_team_id = {team_id} OR g.away_team_id = {team_id})
        """

        # Apply the specific_date filter to the outer query as well
        if specific_date:
            query += f" AND g.official_date = '{specific_date}'"

        query += " ORDER BY subquery.max_end_time DESC"  # Ensure final ordering

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
    MODEL_ID = "gemini-2.0-flash-exp"

    # Structure the prompt to explicitly request tool usage
    structured_prompt = f"""
    You are an expert sports podcast script generator, adept at creating engaging, informative, and dynamic scripts based on user requests and available data. Your task is multifaceted, requiring precise execution across several stages to ensure exceptional output.

    **Overall Goal:** To produce a compelling and meticulously crafted podcast script that accurately addresses user requests, leverages available data effectively, and provides a high-quality listening experience.

    **Step 1: Comprehensive User Request Analysis**
        *   **In-Depth Scrutiny:**  Thoroughly examine the "Question" field, extracting all explicit and implicit requirements. This includes:
            *   **Specificity:** Identify all mentioned teams, players, games (or specific time periods).
            *   **Game Context:** Determine the game type (e.g., regular season, playoffs, exhibition), any specific game focus (key plays, player performance), and critical moments (turning points, upsets).
            *   **Content Focus:** Pinpoint the desired podcast focus (e.g., game analysis, player highlights, team strategy, historical context, record-breaking events).
            *   **Stylistic Preferences:** Understand the desired podcast tone and style (e.g., analytical, enthusiastic, humorous, serious, historical, dramatic).
            *    **Statistical Emphasis:** Identify any specific stats, metrics, or data points the user wants to highlight, including, but not limited to, game dates, final scores, player specific metrics, and any other metrics that provide greater depth to the game. **Crucially, prioritize including all available statistics for mentioned players, teams, and their opponents. This should include, but is not limited to, batting averages, home runs, RBIs, pitching stats (ERA, strikeouts, wins/losses), and fielding statistics. Additionally, be sure to include the names of all starting and key relief pitchers for the game.**
            *   **Implicit Needs:** Infer unspoken requirements based on the question's context (e.g., if a user asks about a close game, anticipate a focus on the final moments).
        *   **Data Prioritization Logic:**  Establish a clear hierarchy for data based on user needs. For example:
            *   Player-centric requests: Prioritize individual player stats, highlights, and pivotal moments.
            *   Game-focused requests: Prioritize game summaries, key events, and strategic plays.
            *   Historical requests: Focus on past game data, trends, records, and historical context.
        *   **Edge Case Management:** Implement robust logic to manage varied user inputs. Specifically:
            *   **Vague Queries:** Develop a fallback strategy for questions like "Tell me about the Lakers." Provide a balanced overview that includes recent games, important historical moments, and significant player performances.
            *   **Conflicting Directives:**  Create a resolution strategy for contradictory requirements (e.g., focus on Player A and Team B). Balance the requests or prioritize based on a logical interpretation of the question. Highlight points where those focus areas intersect in an organic way.
            - **Data Gaps:** If specific game data (e.g., game dates, final scores, **player stats**, , **pitcher information**) is missing, explicitly state in the script that the data was unavailable. Do not use placeholder values. 
            *  **Off-Topic Inquiries:** If the request falls outside the tool's scope (e.g., "What does player X eat"), acknowledge the request is out of scope with a concise message.
            *   **Multiple Entities:** If the user asks for information on multiple teams or players, without specifying a game, provide a summary of their recent performances.
            *  **Aggregated Data:** If the user requests a summary or comparison of multiple players across multiple games, generate an aggregated summary for each player across those games.
            *  **Canceled Events:** If the user requests a game that did not happen, then acknowledge the cancellation.

    **Step 2: Strategic Data Acquisition and Intelligent Analysis**
        *   **Dynamic Tool Selection:** Select the most suitable tool(s) from the available resources based on the refined needs identified in Step 1.  Tools can include statistical APIs, play-by-play logs, news feeds, and social media. Use multiple tools if necessary to gather all the necessary information.
        *  **Prioritized Data Retrieval:** If past games are requested, treat these as primary sources of data and emphasize those data sets. If the user requests a future game or a game with no available data, then state that explicitly in the generated text and use available information like team projections, past performance or other pre game analysis information.
        *   **Granular Data Extraction:** Extract relevant data points, focusing on:
            *   **Critical Events:** Highlight game-changing plays (e.g., game-winning shots, home runs, interceptions).
            *   **Performance Extremes:** Note exceptional performances, unusual dips in performance, or record-breaking accomplishments.
            *   **Pivotal Moments:**  Identify turning points that altered the course of the game.
            *   **Player Insight:** Analyze and report on detailed player actions, individual statistics, and contributions to the game. **Include all relevant stats, such as batting average, home runs, RBIs, and any other available metrics.**
            *   **Game Details:** Extract and include game dates, final scores, and any other relevant game details that add depth and context to the discussion.
            *    **Pitcher Information:** Include starting and key relief pitcher names for each team, as well as their individual stats for the game where available (e.g., innings pitched, strikeouts, earned runs).
        *  **Contextual Layering:** Augment raw data with contextual information to enrich the analysis.
        *  **Contextual Layering:** Augment raw data with contextual information to enrich the analysis.
            *    **Historical Data:** Use past data, historical performance, and historical records, team or player-specific trends to provide the analysis greater depth.
            *    **Team Specific Data:** Use team specific data to better inform the analysis (e.g. if a team is known for strong defense, then analyze this and provide commentary on it).
        *  **Data Integrity Checks:** Sanitize the data to ensure only relevant information is extracted from all sources. Clean and remove any unwanted data.
        * **Edge Case Resolution:** Implement rules for specific edge cases:
            *   **Incomplete Data:** If data is missing or incomplete, explicitly mention this within the generated text using phrases like "data was not available for this event."
            *   **Data Conflicts:** Prioritize reliable sources. If discrepancies persist, note these in the generated text. Explain differences, and any issues that may exist in the data.
            *  **Data Format Issues:**  If the data cannot be parsed or used, then log a detailed error and provide the user with an error in the generated text that explains why data was not used. If possible perform data transformations.

    **Step 3: Advanced Multi-Speaker Script Composition**
        *   **Speaker Profiles:** Develop unique personality profiles for each speaker role to ensure variations in voice and perspective:
             *   **Play-by-play Announcer:** Neutral, factual, and descriptive, providing real-time action updates using clear language.
            *   **Color Commentator:** Analytical, insightful, and contextual, breaking down game elements, offering explanations, and using phrases like "what's interesting here is," "the reason why," and "a key moment in the game".
            *   **Simulated Player Quotes:** Casual, personal, and engaging, re-creating player reactions with plausible, authentic-sounding phrases. **Ensure that for each key play, a simulated player quote is present, that is relevant to the play and provides a unique perspective on the action.**
        *   **Event-Driven Structure:** Structure the script around the key events identified in Step 2. For each event:
             *   Involve all three speaker roles in the conversation to provide multiple perspectives.
            *   Maintain a natural conversation flow, resembling a genuine podcast format.
            *   Incorporate all available relevant information, including player names, team names, inning details, and applicable statistics, **game dates and final scores, and player and pitcher specific stats.**.
        *   **Seamless Transitions:** Use transitional phrases (e.g., "shifting to the next play," "now let's look at the defense") to ensure continuity.
        *   **Unbiased Tone:** Maintain a neutral and factual tone, avoiding any personal opinions, unless specifically instructed by the user.
        *   **Edge Case Handling:**
            *   **Tone Alignment:** Ensure that the speaker's tone reflects the events described (e.g., use a negative tone for the color commentator if describing a poorly executed play).
            *   **Quote Realism:** Ensure simulated quotes are believable and sound authentic.
            *   **Data Gaps:** If there's missing data or an unexpected scenario, use filler phrases (e.g., "We don't have the audio for that play," "Unfortunately, the camera wasn't on the ball").

    **Step 4: Globally Accessible Language Support**
        *   **Translation Integration:** Use translation tools to translate the full output, including all generated text, data-driven content, and speaker roles.
        *   **Language-Specific Adjustments and Chain of Thought Emphasis:**
              - **For Japanese:**  
                   • Use culturally appropriate sports broadcasting language.  
                   • Emphasize the inclusion of the game date and final score by using precise Japanese conventions. 
                   • **Chain-of-Thought:** Begin by clearly stating the game date using Japanese date formats (e.g., "2024年5月15日") and then present the final score using phrases such as "最終スコア." Anchor the entire script in these key details to build a solid factual framework. As you proceed, refer back to these details when transitioning between segments, ensuring that every pivotal play is contextualized within the exact game date and score. This approach not only reinforces the factual basis of the narrative but also resonates with Japanese audiences who expect precision and clarity in sports reporting.
              - **For Spanish:**  
                   • Adopt a lively and engaging commentary style typical of Spanish sports media.  
                   • Stress the inclusion of the game date and final score by using phrases like "la fecha del partido" and "el marcador final" to provide clear factual anchors.  
                   • Chain of Thought: Start the script by emphasizing the importance of the game date using spanish date format and final score, setting the stage for a dynamic narrative. Use vivid descriptions and energetic language to draw the listener into the game, making sure to highlight these key data points repeatedly throughout the script to reinforce the factual context. Detailed descriptions of pivotal plays and smooth transitions will maintain listener engagement while ensuring that the essential facts are always in focus.
              - **For English:**  
                   • Maintain the current detailed and structured narrative with clear emphasis on game dates and final scores as factual anchors.
        *  **Default Language Protocol:** If the user does not specify a language, English will be used as the default language.
        *   **Translation Quality Assurance:** Verify that the translation is accurate and reflects the intended meaning. Ensure that the context of the original text is not lost in translation.
        *   **Edge Case Adaptations:**
            *   **Incomplete Translations:** If the translation is incomplete, use an error code for that section (e.g., `[translation error]`).
            *   **Bidirectional Languages:** Handle languages that read right-to-left to ensure proper text rendering.
           *  **Contextual Accuracy:** Ensure the translation maintains the appropriate tone for the speakers.

    **Step 5: Structured JSON Output Protocol**
        *   **JSON Formatting:** Create the output as a valid JSON array without any additional formatting.
        *   **Speaker and Text Fields:** Each JSON object must include two fields: `"speaker"` and `"text"`.
        *   **Single Array Format:** The output must be a single JSON array containing the entire script.
        *   **No Markdown or Code Blocks:** Do not include any markdown or other formatting elements.
        *   **JSON Validation:** Validate that the output is proper JSON format prior to output.
         *  **Example JSON:**
            ```json
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
            ```
        *   **Edge Case Management:**
            *   **JSON Errors:** If there is a problem creating the json object, then return a json object with an error message.
    **Your Output must be a pure JSON array without any markdown code blocks or formatting. Just the raw JSON.**

    Question: {contents}

    Prioritize the correct execution of each step to ensure the creation of a high-quality, informative, and engaging podcast script, fully tailored to the user's request. Be sure to consider any edge cases in the process.
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
