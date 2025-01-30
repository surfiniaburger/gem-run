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

from google.cloud import bigquery
from google.api_core import exceptions
import logging
from typing import Dict, Union
from datetime import datetime



def fetch_dodgers_games() -> list:
    """
    Fetches all Dodgers games (both home and away) with detailed game information.

    Returns:
        list: A list of dictionaries containing game details including:
              - game_id
              - official_date (in YYYY-MM-DD format)
              - home/away team IDs and names
              - game_type
              - scores
              - venue
              - game status
              - dodgers win/loss and margin

    Raises:
        BigQueryError: If there's an issue with the BigQuery execution.
        Exception: For other unexpected errors.
    """
    try:
        # Define the query
        query = """
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
            dodgers_win,
            dodgers_margin
        FROM
            `gem-rush-007.dodgers_mlb_data_2024.games`
        WHERE
            (home_team_id = 119 OR away_team_id = 119) 
        """
        # Initialize BigQuery client
        bq_client = bigquery.Client()
        
        # Execute the query
        query_job = bq_client.query(query)
        
        # Fetch and process the results with validation
        results = []
        for row in query_job:
            row_dict = dict(row)
            
            # Convert date object to string
            if row_dict.get('official_date'):
                row_dict['official_date'] = row_dict['official_date'].strftime('%Y-%m-%d')
            
            # Validate required fields
            required_fields = [
                'game_id', 'official_date', 'home_team_id', 
                'away_team_id', 'home_team_name', 'away_team_name'
            ]
            
            if not all(row_dict.get(field) for field in required_fields):
                logging.warning(f"Skipping record with missing required information: {row_dict.get('game_id', 'Unknown Game')}")
                continue
            
            # Validate scores
            try:
                if row_dict.get('home_score') is not None:
                    row_dict['home_score'] = int(row_dict['home_score'])
                if row_dict.get('away_score') is not None:
                    row_dict['away_score'] = int(row_dict['away_score'])
            except (TypeError, ValueError) as e:
                logging.warning(f"Invalid score data for game {row_dict['game_id']}: {e}")
                continue
            
            # Validate margin
            if row_dict.get('dodgers_margin') is not None:
                try:
                    row_dict['dodgers_margin'] = int(row_dict['dodgers_margin'])
                except (TypeError, ValueError) as e:
                    logging.warning(f"Invalid margin data for game {row_dict['game_id']}: {e}")
                    continue
            
            results.append(row_dict)
        
        if not results:
            logging.warning("Query returned no results")
            return []
        
        return results
    
    except exceptions.NotFound as e:
        logging.error(f"Table or dataset not found: {e}")
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
        logging.error(f"Unexpected error in fetch_dodgers_games: {e}")
        raise

def fetch_dodgers_player_stats(limit: int = 100) -> list:
    """
    Fetches player statistics from Dodgers games.

    Args:
        limit (int, optional): Maximum number of player records to return. Defaults to 100.

    Returns:
        list: A list of dictionaries containing player statistics including:
              - player_id
              - full_name
              - game_date
              - at_bats
              - hits
              - home_runs
              - rbi
              - walks
              - strikeouts
              - batting_average
              - on_base_percentage
              - slugging_percentage
    """
    try:
        query = """
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
            `gem-rush-007.dodgers_mlb_data_2024.player_stats` AS ps
        JOIN 
            `gem-rush-007.dodgers_mlb_data_2024.roster` AS r
            ON ps.player_id = r.player_id
        INNER JOIN 
            `gem-rush-007.dodgers_mlb_data_2024.games` AS g 
            ON ps.game_id = g.game_id
        WHERE
            (g.home_team_id = 119 OR g.away_team_id = 119)
        ORDER BY 
            g.official_date DESC
        LIMIT @limit
        """

        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("limit", "INT64", limit)
            ]
        )
        
        query_job = bq_client.query(query, job_config=job_config)
        results = list(query_job.result())
        return [dict(row) for row in results]

    except Exception as e:
        logging.error(f"Error in fetch_dodgers_player_stats: {e}")
        return []


def fetch_dodgers_player_stats_by_opponent(opponent_team: str, limit: int = 100) -> list:
    """
    Fetches player statistics from Dodgers games against a specific opponent.

    Args:
        opponent_team (str): The name of the opponent team
        limit (int, optional): Maximum number of records to return. Defaults to 100.

    Returns:
        list: A list of dictionaries containing player statistics
    """
    try:
        query = """
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
            `gem-rush-007.dodgers_mlb_data_2024.player_stats` AS ps
        JOIN 
            `gem-rush-007.dodgers_mlb_data_2024.roster` AS r
            ON ps.player_id = r.player_id
        INNER JOIN 
            `gem-rush-007.dodgers_mlb_data_2024.games` AS g 
            ON ps.game_id = g.game_id
        WHERE
            (g.home_team_id = 119 OR g.away_team_id = 119)
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
        
        query_job = bq_client.query(query, job_config=job_config)
        results = list(query_job.result())
        return [dict(row) for row in results]

    except Exception as e:
        logging.error(f"Error in fetch_dodgers_player_stats_by_opponent: {e}")
        return []

def fetch_dodgers_player_stats_by_game_type(game_type: str, limit: int = 100) -> list:
    """
    Fetches player statistics from Dodgers games of a specific type (R for Regular Season, P for Postseason, etc.).

    Args:
        game_type (str): The type of game (R, P, etc.)
        limit (int, optional): Maximum number of records to return. Defaults to 100.

    Returns:
        list: A list of dictionaries containing player statistics
    """
    try:
        query = """
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
            `gem-rush-007.dodgers_mlb_data_2024.player_stats` AS ps
        JOIN 
            `gem-rush-007.dodgers_mlb_data_2024.roster` AS r
            ON ps.player_id = r.player_id
        INNER JOIN 
            `gem-rush-007.dodgers_mlb_data_2024.games` AS g 
            ON ps.game_id = g.game_id
        WHERE
            (g.home_team_id = 119 OR g.away_team_id = 119)
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
        
        query_job = bq_client.query(query, job_config=job_config)
        results = list(query_job.result())
        return [dict(row) for row in results]

    except Exception as e:
        logging.error(f"Error in fetch_dodgers_player_stats_by_game_type: {e}")
        return []

def fetch_dodgers_plays(limit: int = 100) -> list:
    """
    Fetches plays from all Dodgers games.

    Args:
        limit (int, optional): Maximum number of plays to return. Defaults to 100.

    Returns:
        list: A list of dictionaries containing play details including:
              - play_id
              - inning
              - half_inning
              - event
              - event_type
              - description
              - rbi
              - is_scoring_play
              - batter_name
              - pitcher_name
              - start_time
              - end_time
    """
    try:
        query = """
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
            `gem-rush-007.dodgers_mlb_data_2024.plays` AS p
        LEFT JOIN 
            `gem-rush-007.dodgers_mlb_data_2024.roster` as r_batter 
            ON p.batter_id = r_batter.player_id
        LEFT JOIN 
            `gem-rush-007.dodgers_mlb_data_2024.roster` as r_pitcher 
            ON p.pitcher_id = r_pitcher.player_id
        INNER JOIN 
            `gem-rush-007.dodgers_mlb_data_2024.games` AS g 
            ON p.game_id = g.game_id
        WHERE
            g.home_team_id = 119 OR g.away_team_id = 119
        ORDER BY 
            g.official_date DESC, p.start_time ASC
        LIMIT @limit
        """

        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("limit", "INT64", limit)
            ]
        )
        
        query_job = bq_client.query(query, job_config=job_config)
        results = list(query_job.result())
        return [dict(row) for row in results]

    except Exception as e:
        logging.error(f"Error in fetch_dodgers_plays: {e}")
        return []



def fetch_dodgers_plays_by_opponent(opponent_team: str, limit: int = 100) -> list:
    """
    Fetches plays from Dodgers games against a specific opponent.

    Args:
        opponent_team (str): The name of the opponent team
        limit (int, optional): Maximum number of plays to return. Defaults to 100.

    Returns:
        list: A list of dictionaries containing play details including:
              - play_id
              - inning
              - half_inning
              - event
              - event_type
              - description
              - rbi
              - is_scoring_play
              - batter_name
              - pitcher_name
              - start_time
              - end_time
    """
    try:
        query = """
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
            `gem-rush-007.dodgers_mlb_data_2024.plays` AS p
        LEFT JOIN 
            `gem-rush-007.dodgers_mlb_data_2024.roster` as r_batter 
            ON p.batter_id = r_batter.player_id
        LEFT JOIN 
            `gem-rush-007.dodgers_mlb_data_2024.roster` as r_pitcher 
            ON p.pitcher_id = r_pitcher.player_id
        INNER JOIN 
            `gem-rush-007.dodgers_mlb_data_2024.games` AS g 
            ON p.game_id = g.game_id
        WHERE
            (g.home_team_id = 119 OR g.away_team_id = 119)
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
        
        query_job = bq_client.query(query, job_config=job_config)
        results = list(query_job.result())
        return [dict(row) for row in results]

    except Exception as e:
        logging.error(f"Error in fetch_dodgers_plays_by_opponent: {e}")
        return []

def fetch_dodgers_plays_by_game_type(game_type: str, limit: int = 100) -> list:
    """
    Fetches plays from Dodgers games of a specific type (R for Regular Season, P for Postseason, etc.).

    Args:
        game_type (str): The type of game (R, P, etc.)
        limit (int, optional): Maximum number of plays to return. Defaults to 100.

    Returns:
        list: A list of dictionaries containing play details including:
              - play_id
              - inning
              - half_inning
              - event
              - event_type
              - description
              - rbi
              - is_scoring_play
              - batter_name
              - pitcher_name
              - start_time
              - end_time
    """
    try:
        query = """
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
            `gem-rush-007.dodgers_mlb_data_2024.plays` AS p
        LEFT JOIN 
            `gem-rush-007.dodgers_mlb_data_2024.roster` as r_batter 
            ON p.batter_id = r_batter.player_id
        LEFT JOIN 
            `gem-rush-007.dodgers_mlb_data_2024.roster` as r_pitcher 
            ON p.pitcher_id = r_pitcher.player_id
        INNER JOIN 
            `gem-rush-007.dodgers_mlb_data_2024.games` AS g 
            ON p.game_id = g.game_id
        WHERE
            (g.home_team_id = 119 OR g.away_team_id = 119)
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
        
        query_job = bq_client.query(query, job_config=job_config)
        results = list(query_job.result())
        return [dict(row) for row in results]

    except Exception as e:
        logging.error(f"Error in fetch_dodgers_plays_by_game_type: {e}")
        return []



def fetch_dodgers_games_by_opponent(opponent_team: str = 'New York Yankees', limit: int = 2) -> list:
    """
    Fetches Dodgers games against a specific opponent.

    Args:
        opponent_team (str, optional): Name of the opponent team. Defaults to 'New York Yankees'.
        limit (int, optional): Number of most recent games to return. Defaults to 2.

    Returns:
        list: A list of dictionaries containing game information including:
              - game_id
              - official_date
              - home_team_id
              - home_team_name
              - away_team_id
              - away_team_name
              - home_score
              - away_score
              - venue_name
              - status
              - dodgers_win
              - dodgers_margin

    Raises:
        ValueError: If invalid parameters are provided.
        BigQueryError: If there's an issue with the BigQuery execution.
        Exception: For other unexpected errors.
    """
    try:
        # Input validation
        if not isinstance(limit, int) or limit <= 0:
            raise ValueError("limit must be a positive integer")
        if not isinstance(opponent_team, str) or not opponent_team.strip():
            raise ValueError("opponent_team must be a non-empty string")

        # Build the query
        query = """
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
            dodgers_win,
            dodgers_margin
        FROM
            `gem-rush-007.dodgers_mlb_data_2024.games`
        WHERE
            (home_team_id = 119 OR away_team_id = 119)
            AND ((home_team_name = @opponent_team) OR (away_team_name = @opponent_team))
        ORDER BY official_date DESC
        LIMIT @limit
        """

        # Set up query parameters
        query_params = [
            bigquery.ScalarQueryParameter("limit", "INT64", limit),
            bigquery.ScalarQueryParameter("opponent_team", "STRING", opponent_team)
        ]

        # Configure and execute query
        job_config = bigquery.QueryJobConfig(query_parameters=query_params)
        bq_client = bigquery.Client()
        query_job = bq_client.query(query, job_config=job_config)
        
        return process_game_results(query_job)

    except Exception as e:
        logging.error(f"Unexpected error in fetch_dodgers_games_by_opponent: {e}")
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
        logging.error(f"Unexpected error in fetch_dodgers_games: {e}")
        raise

def fetch_dodgers_games_by_type(game_type: str = 'R', limit: int = 2) -> list:
    """
    Fetches Dodgers games by game type.

    Args:
        game_type (str, optional): Type of game ('R' for Regular Season, 'L' for League Championship, 
                                 'D' for Division Series, 'W' for World Series, etc.). 
                                 Defaults to 'R'.
        limit (int, optional): Number of most recent games to return. Defaults to 2.

    Returns:
        list: A list of dictionaries containing game information including:
              - game_id
              - official_date
              - home_team_id
              - home_team_name
              - away_team_id
              - away_team_name
              - home_score
              - away_score
              - venue_name
              - status
              - dodgers_win
              - dodgers_margin

    Raises:
        ValueError: If invalid parameters are provided.
        BigQueryError: If there's an issue with the BigQuery execution.
        Exception: For other unexpected errors.
    """
    try:
        # Input validation
        if not isinstance(limit, int) or limit <= 0:
            raise ValueError("limit must be a positive integer")
        if not isinstance(game_type, str) or not game_type.strip():
            raise ValueError("game_type must be a non-empty string")

        # Build the query
        query = """
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
            dodgers_win,
            dodgers_margin
        FROM
            `gem-rush-007.dodgers_mlb_data_2024.games`
        WHERE
            (home_team_id = 119 OR away_team_id = 119)
            AND game_type = @game_type
        ORDER BY official_date DESC
        LIMIT @limit
        """

        # Set up query parameters
        query_params = [
            bigquery.ScalarQueryParameter("limit", "INT64", limit),
            bigquery.ScalarQueryParameter("game_type", "STRING", game_type)
        ]

        # Configure and execute query
        job_config = bigquery.QueryJobConfig(query_parameters=query_params)
        bq_client = bigquery.Client()
        query_job = bq_client.query(query, job_config=job_config)
        
        return process_game_results(query_job)

    except Exception as e:
        logging.error(f"Unexpected error in fetch_dodgers_games_by_type: {e}")
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
        logging.error(f"Unexpected error in fetch_dodgers_games: {e}")
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
    


def fetch_player_game_stats(game_ids: list = None, player_ids: list = None) -> list:
    """
    Fetches player statistics for specified games and/or players.

    Args:
        game_ids (list, optional): List of game IDs to filter by. If None, no game filtering.
        player_ids (list, optional): List of player IDs to filter by. If None, all players included.

    Returns:
        list: A list of dictionaries containing player stats including:
              - player_id
              - full_name
              - at_bats
              - hits
              - home_runs
              - rbi
              - walks
              - strikeouts
              - batting_average
              - on_base_percentage
              - slugging_percentage

    Raises:
        ValueError: If invalid parameters are provided.
        BigQueryError: If there's an issue with the BigQuery execution.
        Exception: For other unexpected errors.
    """
    try:
        # Input validation
        if game_ids is not None and not isinstance(game_ids, list):
            raise ValueError("game_ids must be a list or None")
        if player_ids is not None and not isinstance(player_ids, list):
            raise ValueError("player_ids must be a list or None")

        # Build the base query
        query = """
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
            `gem-rush-007.dodgers_mlb_data_2024.player_stats` AS ps
        JOIN 
            `gem-rush-007.dodgers_mlb_data_2024.roster` AS r
        ON 
            ps.player_id = r.player_id
        """


        # Initialize parameters list and WHERE conditions
        query_params = []
        where_conditions = []

        # Add game_id filter if provided
        if game_ids:
            where_conditions.append("ps.game_id IN UNNEST(@game_id_list)")
            query_params.append(
                bigquery.ArrayQueryParameter("game_id_list", "STRING", game_ids)
            )

        # Add player_id filter if provided
        if player_ids:
            where_conditions.append("ps.player_id IN UNNEST(@player_id_list)")
            query_params.append(
                bigquery.ArrayQueryParameter("player_id_list", "STRING", player_ids)
            )

        # Add WHERE clause if there are any conditions
        if where_conditions:
            query += "\nWHERE " + " AND ".join(where_conditions)
        # Configure the query
        job_config = bigquery.QueryJobConfig(query_parameters=query_params)
        
        # Initialize BigQuery client
        bq_client = bigquery.Client()
        
        # Execute the query
        query_job = bq_client.query(query, job_config=job_config)
        
        # Fetch and process the results with validation
        results = []
        for row in query_job:
            row_dict = dict(row)
            
            # Validate required fields
            required_fields = ['player_id', 'full_name']
            if not all(row_dict.get(field) for field in required_fields):
                logging.warning(f"Skipping record with missing required information: {row_dict.get('player_id', 'Unknown Player')}")
                continue
            
            # Validate numeric fields
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
            logging.warning("Query returned no results")
            return []
        
        return results
    
    except exceptions.NotFound as e:
        logging.error(f"Table or dataset not found: {e}")
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
        logging.error(f"Unexpected error in fetch_player_game_stats: {e}")
        raise




def fetch_player_plays(player_name: str, limit: int = 100) -> list:

    """
    Fetches play-by-play data for a specific player from Dodgers games and generates a Looker Studio iframe URL.

    Args:
        player_name (str): Full name of the player
        limit (int, optional): Maximum number of plays to return. Defaults to 100.

    Returns:
        list: A list dictionary containing:
            - plays (list): List of dictionaries containing play-by-play data
    """
    try:
        query = """
        SELECT
            p.play_id,
            p.inning,
            p.half_inning,
            p.event,
            p.event_type,
            p.description,
            p.start_time,
            g.official_date as game_date,
         
        FROM
            `gem-rush-007.dodgers_mlb_data_2024.plays` AS p
        INNER JOIN 
            `gem-rush-007.dodgers_mlb_data_2024.games` AS g 
            ON p.game_id = g.game_id
        INNER JOIN
            `gem-rush-007.dodgers_mlb_data_2024.roster` AS r
            ON (p.batter_id = r.player_id OR p.pitcher_id = r.player_id)
        
        WHERE
            r.full_name = @player_name
            AND (g.home_team_id = 119 OR g.away_team_id = 119)
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
        
        query_job = bq_client.query(query, job_config=job_config)
        results = list(query_job.result())

        # Convert the results to dictionaries and format datetime objects
        formatted_results = []
        for row in results:
            row_dict = dict(row)
            # Convert datetime objects to ISO format strings
            if 'start_time' in row_dict and row_dict['start_time']:
                row_dict['start_time'] = row_dict['start_time'].isoformat()
            if 'game_date' in row_dict and row_dict['game_date']:
                row_dict['game_date'] = row_dict['game_date'].isoformat()
            formatted_results.append(row_dict)
        return formatted_results
      
    except Exception as e:
        logging.error(f"Error in fetch_player_plays: {e}")
        return []

def fetch_player_plays_by_opponent(player_name: str, opponent_team: str, limit: int = 100) -> list:
    """
    Fetches play-by-play data for a specific player against a specific opponent.

    Args:
        player_name (str): Full name of the player
        opponent_team (str): Name of the opponent team
        limit (int, optional): Maximum number of plays to return. Defaults to 100.

    Returns:
        list: A list of dictionaries containing play-by-play data
    """
    try:
        query = """
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
            `gem-rush-007.dodgers_mlb_data_2024.plays` AS p
        INNER JOIN 
            `gem-rush-007.dodgers_mlb_data_2024.games` AS g 
            ON p.game_id = g.game_id
        INNER JOIN
            `gem-rush-007.dodgers_mlb_data_2024.roster` AS r
            ON p.batter_id = r.player_id
        WHERE
            r.full_name = @player_name
            AND (g.home_team_id = 119 OR g.away_team_id = 119)
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
        
        query_job = bq_client.query(query, job_config=job_config)
        results = list(query_job.result())
        return [dict(row) for row in results]

    except Exception as e:
        logging.error(f"Error in fetch_player_plays_by_opponent: {e}")
        return []

def fetch_player_plays_by_game_type(player_name: str, game_type: str, limit: int = 100) -> list:
    """
    Fetches play-by-play data for a specific player from games of a specific type.

    Args:
        player_name (str): Full name of the player
        game_type (str): Type of game (R for Regular Season, P for Postseason, etc.)
        limit (int, optional): Maximum number of plays to return. Defaults to 100.

    Returns:
        list: A list of dictionaries containing play-by-play data
    """
    try:
        query = """
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
            `gem-rush-007.dodgers_mlb_data_2024.plays` AS p
        INNER JOIN 
            `gem-rush-007.dodgers_mlb_data_2024.games` AS g 
            ON p.game_id = g.game_id
        INNER JOIN
            `gem-rush-007.dodgers_mlb_data_2024.roster` AS r
            ON p.batter_id = r.player_id
        WHERE
            r.full_name = @player_name
            AND (g.home_team_id = 119 OR g.away_team_id = 119)
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
        
        query_job = bq_client.query(query, job_config=job_config)
        results = list(query_job.result())
        return [dict(row) for row in results]

    except Exception as e:
        logging.error(f"Error in fetch_player_plays_by_game_type: {e}")
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
                    # Prioritize the player_plays function
                    fetch_player_plays,
                    fetch_dodgers_games,
                    fetch_dodgers_games_by_opponent,
                    fetch_dodgers_games_by_type,
                    fetch_dodgers_plays,
                    fetch_dodgers_plays_by_opponent,
                    fetch_dodgers_plays_by_game_type,
                    fetch_dodgers_player_stats,
                    fetch_dodgers_player_stats_by_game_type,
                    fetch_dodgers_player_stats_by_opponent,
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