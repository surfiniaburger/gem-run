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
from typing import List, Dict, Union
from datetime import datetime

logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


PROJECT_ID = "gem-creation"  # Replace with your actual Google Cloud project ID
os.environ["GOOGLE_CLOUD_PROJECT"] = PROJECT_ID # set this environment variable to your project ID
bq_client = bigquery.Client(project=PROJECT_ID)

client = genai.Client(vertexai=True, project="gem-creation", location="us-central1")
MODEL_ID = "gemini-2.0-flash-exp"  # @param {type: "string"}

from google.cloud import bigquery
from google.api_core import exceptions
import logging
from typing import Dict, Union
from datetime import datetime

def get_player_highest_ops(season: int) -> Dict[str, Union[str, float]]:
    """Returns the player with the highest OPS (On-base Plus Slugging) for a given season.

    Args:
        season: The MLB season year (e.g., 2020)
    
    Returns:
        Dict: Player information including name, OPS, and team
    
    Raises:
        ValueError: If season parameter is invalid
        IndexError: If no data is found for the given season
        BigQueryError: If there's an issue with the BigQuery execution
        Exception: For other unexpected errors
    """
    # Input validation
    current_year = datetime.now().year
    
    if not isinstance(season, int):
        raise ValueError("Season must be an integer")
        
    if season < 1876 or season > current_year:
        raise ValueError(f"Season must be between 1876 and {current_year}")

    try:
        query = """
        SELECT
            first_name,
            last_name,
            on_base_plus_slugging,
            team_name,
            games_played,
            at_bats
        FROM
            `mlb_data.combined_player_stats`
        WHERE
            season = @season
            AND on_base_plus_slugging IS NOT NULL
            AND on_base_plus_slugging >= 0
            AND games_played >= 10  -- Minimum games threshold
            AND at_bats >= 30      -- Minimum at-bats threshold
        QUALIFY
            ROW_NUMBER() OVER (PARTITION BY season 
                             ORDER BY on_base_plus_slugging DESC) = 1
        """

        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("season", "INTEGER", season)
            ]
        )

        # Execute query with timeout
        query_job = bq_client.query(query, job_config=job_config)
        results = list(query_job.result(timeout=30))  # 30 second timeout
        
        if not results:
            logging.warning(f"No qualified players found for season {season}")
            raise IndexError(f"No qualified players found for season {season}")
            
        result_dict = dict(results[0])
        
        # Validate required fields
        required_fields = ['first_name', 'last_name', 'on_base_plus_slugging', 'team_name']
        if not all(field in result_dict for field in required_fields):
            logging.error("Missing required fields in query result")
            raise ValueError("Incomplete player data returned from query")
            
        # Validate OPS value
        if not (0 <= result_dict['on_base_plus_slugging'] <= 5.0):  # 5.0 is an extremely generous upper limit
            logging.warning(f"Suspicious OPS value ({result_dict['on_base_plus_slugging']}) "
                          f"for {result_dict['first_name']} {result_dict['last_name']}")
            
        # Validate games played and at-bats
        if result_dict['games_played'] < 10:
            logging.warning("Player has fewer than 10 games played")
            
        if result_dict['at_bats'] < 30:
            logging.warning("Player has fewer than 30 at-bats")
            
        # Remove extra fields before returning
        for field in ['games_played', 'at_bats']:
            result_dict.pop(field, None)
            
        # Round OPS to 3 decimal places
        result_dict['on_base_plus_slugging'] = round(result_dict['on_base_plus_slugging'], 3)
            
        return result_dict

    except IndexError as e:
        logging.error(f"No data found for season {season}")
        raise

    except exceptions.Timeout as e:
        logging.error(f"Query timed out: {e}")
        raise

    except exceptions.NotFound as e:
        logging.error(f"Table or dataset not found: {e}")
        raise

    except exceptions.BadRequest as e:
        logging.error(f"Invalid query or bad request: {e}")
        raise

    except exceptions.Forbidden as e:
        logging.error(f"Permission denied accessing BigQuery: {e}")
        raise

    except exceptions.GoogleAPIError as e:
        logging.error(f"BigQuery API error: {e}")
        raise

    except Exception as e:
        logging.error(f"Unexpected error in get_player_highest_ops: {e}")
        raise


def calculate_team_home_advantage(team_name: str) -> Dict[str, Union[str, float]]:
    """Calculates the home field advantage statistics for a specific team.
    
    Args:
        team_name: The name of the MLB team
    
    Returns:
        Dict: Analysis of the team's home field advantage including win percentage and scoring averages
    
    Raises:
        ValueError: If team_name is invalid
        IndexError: If no data is found for the given team
        BigQueryError: If there's an issue with the BigQuery execution
        Exception: For other unexpected errors
    """
    # Input validation
    if not isinstance(team_name, str):
        raise ValueError("Team name must be a string")
        
    if not team_name.strip():
        raise ValueError("Team name cannot be empty")

    try:
        query = """
        WITH GameStats AS (
            SELECT
                team_name,
                home_score,
                away_score,
                COUNT(*) OVER () as total_games
            FROM
                `mlb_data.combined_player_stats`
            WHERE
                team_name = @team_name
                AND home_score IS NOT NULL
                AND away_score IS NOT NULL
                AND home_score >= 0
                AND away_score >= 0
        )
        SELECT
            team_name,
            COUNTIF(home_score > away_score) * 100.0 / COUNT(*) AS home_win_percentage,
            AVG(home_score) AS avg_home_runs,
            AVG(away_score) AS avg_away_runs,
            COUNT(*) as games_analyzed,
            COUNTIF(home_score > away_score) as home_wins,
            COUNTIF(home_score < away_score) as home_losses,
            COUNTIF(home_score = away_score) as home_ties
        FROM
            GameStats
        GROUP BY
            team_name
        HAVING
            COUNT(*) >= 10  -- Minimum games threshold
        """

        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("team_name", "STRING", team_name)
            ]
        )

        # Execute query with timeout
        query_job = bq_client.query(query, job_config=job_config)
        results = list(query_job.result(timeout=30))  # 30 second timeout
        
        if not results:
            logging.warning(f"No data found for team: {team_name}")
            raise IndexError(f"No data found for team: {team_name}")
            
        result_dict = dict(results[0])
        
        # Validate required fields
        required_fields = ['team_name', 'home_win_percentage', 'avg_home_runs', 
                         'avg_away_runs', 'games_analyzed']
        if not all(field in result_dict for field in required_fields):
            logging.error("Missing required fields in query result")
            raise ValueError("Incomplete team data returned from query")
            
        # Validate numeric values
        if result_dict['games_analyzed'] < 10:
            logging.warning(f"Limited sample size: only {result_dict['games_analyzed']} games analyzed")
            
        if not (0 <= result_dict['home_win_percentage'] <= 100):
            logging.error(f"Invalid win percentage: {result_dict['home_win_percentage']}")
            raise ValueError("Invalid win percentage calculated")
            
        if (result_dict['avg_home_runs'] < 0 or 
            result_dict['avg_away_runs'] < 0 or 
            result_dict['avg_home_runs'] > 30 or  # Reasonable upper limit for average runs
            result_dict['avg_away_runs'] > 30):
            logging.warning("Suspicious average run values detected")
            
        # Round numeric values for consistency
        result_dict['home_win_percentage'] = round(result_dict['home_win_percentage'], 2)
        result_dict['avg_home_runs'] = round(result_dict['avg_home_runs'], 2)
        result_dict['avg_away_runs'] = round(result_dict['avg_away_runs'], 2)
        
        # Calculate home field advantage
        result_dict['run_differential'] = round(result_dict['avg_home_runs'] - result_dict['avg_away_runs'], 2)
        
        # Remove unnecessary fields
        for field in ['home_ties', 'games_analyzed']:
            result_dict.pop(field, None)
            
        return result_dict

    except IndexError as e:
        logging.error(f"No data found for team {team_name}")
        raise

    except exceptions.Timeout as e:
        logging.error(f"Query timed out: {e}")
        raise

    except exceptions.NotFound as e:
        logging.error(f"Table or dataset not found: {e}")
        raise

    except exceptions.BadRequest as e:
        logging.error(f"Invalid query or bad request: {e}")
        raise

    except exceptions.Forbidden as e:
        logging.error(f"Permission denied accessing BigQuery: {e}")
        raise

    except exceptions.GoogleAPIError as e:
        logging.error(f"BigQuery API error: {e}")
        raise

    except Exception as e:
        logging.error(f"Unexpected error in calculate_team_home_advantage: {e}")
        raise

def analyze_player_performance(player_first_name: str, player_last_name: str) -> Dict[str, Union[str, float, int]]:
    """Analyzes a player's career statistics and performance trends.
    
    Args:
        player_first_name: Player's first name
        player_last_name: Player's last name
    
    Returns:
        Dict: Comprehensive analysis of the player's performance
    
    Raises:
        ValueError: If player name parameters are invalid
        IndexError: If no data is found for the player
        BigQueryError: If there's an issue with the BigQuery execution
        Exception: For other unexpected errors
    """
    # Input validation
    if not isinstance(player_first_name, str) or not isinstance(player_last_name, str):
        raise ValueError("Player names must be strings")
        
    if not player_first_name.strip() or not player_last_name.strip():
        raise ValueError("Player names cannot be empty")
        
    # Remove extra whitespace and capitalize names
    player_first_name = player_first_name.strip().title()
    player_last_name = player_last_name.strip().title()

    try:
        query = """
        WITH PlayerStats AS (
            SELECT
                first_name,
                last_name,
                season,
                batting_average,
                homeruns,
                stolen_bases,
                games_played,
                at_bats
            FROM
                `mlb_data.combined_player_stats`
            WHERE
                first_name = @player_first_name
                AND last_name = @player_last_name
                AND batting_average IS NOT NULL
                AND batting_average >= 0
                AND batting_average <= 1
                AND games_played > 0
        )
        SELECT
            first_name,
            last_name,
            AVG(batting_average) as career_avg,
            SUM(homeruns) as total_hr,
            SUM(stolen_bases) as total_sb,
            COUNT(DISTINCT season) as seasons_played,
            SUM(games_played) as total_games,
            SUM(at_bats) as total_at_bats,
            MIN(season) as first_season,
            MAX(season) as last_season
        FROM
            PlayerStats
        GROUP BY
            first_name,
            last_name
        HAVING
            COUNT(DISTINCT season) > 0
        """

        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("player_first_name", "STRING", player_first_name),
                bigquery.ScalarQueryParameter("player_last_name", "STRING", player_last_name)
            ]
        )

        # Execute query with timeout
        query_job = bq_client.query(query, job_config=job_config)
        results = list(query_job.result(timeout=30))  # 30 second timeout
        
        if not results:
            logging.warning(f"No data found for player: {player_first_name} {player_last_name}")
            raise IndexError(f"No data found for player: {player_first_name} {player_last_name}")
            
        result_dict = dict(results[0])
        
        # Validate required fields
        required_fields = ['career_avg', 'total_hr', 'total_sb', 'seasons_played', 
                         'total_games', 'total_at_bats']
        if not all(field in result_dict for field in required_fields):
            logging.error("Missing required fields in query result")
            raise ValueError("Incomplete player data returned from query")
            
        # Validate numeric values
        if result_dict['seasons_played'] <= 0:
            logging.error("Invalid seasons played count")
            raise ValueError("Invalid seasons played count")
            
        if not (0 <= result_dict['career_avg'] <= 1):
            logging.warning(f"Suspicious career average: {result_dict['career_avg']}")
            
        if result_dict['total_hr'] < 0 or result_dict['total_sb'] < 0:
            logging.error("Negative statistics detected")
            raise ValueError("Invalid negative statistics found")
            
        # Calculate additional statistics
        result_dict['games_per_season'] = round(result_dict['total_games'] / result_dict['seasons_played'], 1)
        result_dict['career_length'] = result_dict['last_season'] - result_dict['first_season'] + 1
        
        # Round numeric values for consistency
        result_dict['career_avg'] = round(result_dict['career_avg'], 3)
        
        # Remove unnecessary fields
        for field in ['first_name', 'last_name']:
            result_dict.pop(field, None)
            
        return result_dict

    except IndexError as e:
        logging.error(f"No data found for player {player_first_name} {player_last_name}")
        raise

    except exceptions.Timeout as e:
        logging.error(f"Query timed out: {e}")
        raise

    except exceptions.NotFound as e:
        logging.error(f"Table or dataset not found: {e}")
        raise

    except exceptions.BadRequest as e:
        logging.error(f"Invalid query or bad request: {e}")
        raise

    except exceptions.Forbidden as e:
        logging.error(f"Permission denied accessing BigQuery: {e}")
        raise

    except exceptions.GoogleAPIError as e:
        logging.error(f"BigQuery API error: {e}")
        raise

    except Exception as e:
        logging.error(f"Unexpected error in analyze_player_performance: {e}")
        raise

def analyze_weight_performance(min_weight: int, max_weight: int) -> Dict[str, Union[float, int]]:
    """Analyzes batting performance statistics for players within a weight range.
    
    Args:
        min_weight: Minimum player weight
        max_weight: Maximum player weight
    
    Returns:
        Dict: Batting performance statistics including averages and player count
    
    Raises:
        ValueError: If weight parameters are invalid
        IndexError: If no data is found for the weight range
        BigQueryError: If there's an issue with the BigQuery execution
        Exception: For other unexpected errors
    """
    # Input validation
    if not isinstance(min_weight, int) or not isinstance(max_weight, int):
        raise ValueError("Weight values must be integers")
        
    if min_weight < 100 or max_weight > 400:  # Reasonable MLB player weight range
        raise ValueError("Weight must be between 100 and 400 pounds")
        
    if min_weight > max_weight:
        raise ValueError("Minimum weight must be less than maximum weight")
        
    if max_weight - min_weight < 10:
        raise ValueError("Weight range must be at least 10 pounds")

    try:
        query = """
        WITH PlayerStats AS (
            SELECT
                player_id,
                weight,
                batting_average,
                homeruns,
                at_bats,
                games_played
            FROM
                `mlb_data.combined_player_stats`
            WHERE
                weight BETWEEN @min_weight AND @max_weight
                AND weight IS NOT NULL
                AND batting_average IS NOT NULL
                AND batting_average BETWEEN 0 AND 1
                AND at_bats >= 200
                AND games_played >= 20
        )
        SELECT
            AVG(batting_average) AS avg_batting_average,
            AVG(homeruns) AS avg_homeruns,
            COUNT(DISTINCT player_id) AS player_count,
            MIN(weight) AS min_weight_found,
            MAX(weight) AS max_weight_found,
            AVG(weight) AS avg_weight,
            MIN(batting_average) AS min_batting_avg,
            MAX(batting_average) AS max_batting_avg
        FROM
            PlayerStats
        HAVING
            COUNT(DISTINCT player_id) > 0
        """

        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("min_weight", "INT64", min_weight),
                bigquery.ScalarQueryParameter("max_weight", "INT64", max_weight)
            ]
        )

        # Execute query with timeout
        query_job = bq_client.query(query, job_config=job_config)
        results = list(query_job.result(timeout=30))  # 30 second timeout
        
        if not results:
            logging.warning(f"No data found for weight range {min_weight}-{max_weight}")
            raise IndexError(f"No data found for weight range {min_weight}-{max_weight}")
            
        result_dict = dict(results[0])
        
        # Validate required fields
        required_fields = ['avg_batting_average', 'avg_homeruns', 'player_count']
        if not all(field in result_dict for field in required_fields):
            logging.error("Missing required fields in query result")
            raise ValueError("Incomplete data returned from query")
            
        # Validate player count
        if result_dict['player_count'] <= 0:
            logging.error("No qualifying players found")
            raise ValueError("No qualifying players found in the specified weight range")
            
        # Validate numeric values
        if not (0 <= result_dict['avg_batting_average'] <= 1):
            logging.warning(f"Suspicious average batting average: {result_dict['avg_batting_average']}")
            
        if result_dict['avg_homeruns'] < 0:
            logging.error("Negative home run average detected")
            raise ValueError("Invalid negative statistics found")
            
        # Add derived statistics
        result_dict['weight_range'] = max_weight - min_weight
        
        # Round numeric values for consistency
        result_dict['avg_batting_average'] = round(result_dict['avg_batting_average'], 3)
        result_dict['avg_homeruns'] = round(result_dict['avg_homeruns'], 1)
        result_dict['avg_weight'] = round(result_dict.get('avg_weight', 0), 1)
        
        # Remove unnecessary fields
        for field in ['min_weight_found', 'max_weight_found']:
            result_dict.pop(field, None)
            
        return result_dict

    except IndexError as e:
        logging.error(f"No data found for weight range {min_weight}-{max_weight}")
        raise

    except exceptions.Timeout as e:
        logging.error(f"Query timed out: {e}")
        raise

    except exceptions.NotFound as e:
        logging.error(f"Table or dataset not found: {e}")
        raise

    except exceptions.BadRequest as e:
        logging.error(f"Invalid query or bad request: {e}")
        raise

    except exceptions.Forbidden as e:
        logging.error(f"Permission denied accessing BigQuery: {e}")
        raise

    except exceptions.GoogleAPIError as e:
        logging.error(f"BigQuery API error: {e}")
        raise

    except Exception as e:
        logging.error(f"Unexpected error in analyze_weight_performance: {e}")
        raise


def analyze_team_strikeouts(team_name: str, season: int) -> Dict[str, Union[str, float, int]]:
    """Analyzes team's strikeout statistics for a specific season.
    
    Args:
        team_name: The name of the MLB team
        season: The season year to analyze
    
    Returns:
        Dict: Team strikeout statistics
    
    Raises:
        ValueError: If input parameters are invalid
        IndexError: If no data is found
        BigQueryError: If there's an issue with the BigQuery execution
    """
    # Input validation
    if not isinstance(team_name, str) or not team_name.strip():
        raise ValueError("Team name must be a non-empty string")
        
    if not isinstance(season, int):
        raise ValueError("Season must be an integer")
        
    current_year = datetime.now().year
    if season < 1876 or season > current_year:
        raise ValueError(f"Season must be between 1876 and {current_year}")

    try:
        query = """
        SELECT
            team_name,
            AVG(strikeouts) AS average_strikeouts_per_game,
            MAX(strikeouts) AS max_strikeouts,
            MIN(strikeouts) AS min_strikeouts,
            COUNT(DISTINCT game_pk) as games_played,
            SUM(strikeouts) as total_strikeouts
        FROM
            `mlb_data.combined_player_stats`
        WHERE
            team_name = @team_name
            AND season = @season
            AND strikeouts IS NOT NULL
            AND strikeouts >= 0
        GROUP BY
            team_name
        HAVING
            COUNT(DISTINCT game_pk) >= 10
        """

        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("team_name", "STRING", team_name.strip()),
                bigquery.ScalarQueryParameter("season", "INT64", season)
            ]
        )

        query_job = bq_client.query(query, job_config=job_config)
        results = list(query_job.result(timeout=30))

        if not results:
            raise IndexError(f"No data found for {team_name} in {season}")

        result_dict = dict(results[0])
        
        # Validate results
        if result_dict['games_played'] < 10:
            raise ValueError(f"Insufficient games ({result_dict['games_played']}) for analysis")

        # Round values for consistency
        result_dict['average_strikeouts_per_game'] = round(result_dict['average_strikeouts_per_game'], 2)
        
        return result_dict

    except (exceptions.Timeout, exceptions.GoogleAPIError) as e:
        logging.error(f"BigQuery error: {str(e)}")
        raise
    except Exception as e:
        logging.error(f"Error in analyze_team_strikeouts: {str(e)}")
        raise

def analyze_monthly_home_runs(team_name: str, year: int) -> List[Dict[str, Union[str, int, float]]]:
    """Analyzes monthly home run trends for a specific team and year.
    
    Args:
        team_name: The name of the MLB team
        year: The year to analyze
    
    Returns:
        List[Dict]: Monthly home run statistics
    
    Raises:
        ValueError: If input parameters are invalid
        IndexError: If no data is found
        BigQueryError: If there's an issue with the BigQuery execution
    """
    # Input validation
    if not isinstance(team_name, str) or not team_name.strip():
        raise ValueError("Team name must be a non-empty string")
        
    if not isinstance(year, int):
        raise ValueError("Year must be an integer")
        
    current_year = datetime.now().year
    if year < 1876 or year > current_year:
        raise ValueError(f"Year must be between 1876 and {current_year}")

    try:
        query = """
        WITH MonthlyStats AS (
            SELECT
                team_name,
                EXTRACT(MONTH FROM game_date) AS month,
                AVG(homeruns) AS average_home_runs,
                COUNT(DISTINCT game_pk) as games_played,
                SUM(homeruns) as total_home_runs
            FROM
                `mlb_data.combined_player_stats`
            WHERE
                team_name = @team_name
                AND EXTRACT(YEAR FROM game_date) = @year
                AND homeruns IS NOT NULL
                AND homeruns >= 0
            GROUP BY
                team_name,
                month
            HAVING
                COUNT(DISTINCT game_pk) >= 5
        )
        SELECT 
            *,
            total_home_runs / NULLIF(games_played, 0) as home_runs_per_game
        FROM MonthlyStats
        ORDER BY
            month
        """

        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("team_name", "STRING", team_name.strip()),
                bigquery.ScalarQueryParameter("year", "INT64", year)
            ]
        )

        query_job = bq_client.query(query, job_config=job_config)
        results = list(query_job.result(timeout=30))

        if not results:
            raise IndexError(f"No data found for {team_name} in {year}")

        processed_results = []
        for row in results:
            result_dict = dict(row)
            
            # Validate month
            if not (1 <= result_dict['month'] <= 12):
                logging.warning(f"Invalid month {result_dict['month']} found")
                continue
                
            # Validate games played
            if result_dict['games_played'] < 5:
                logging.warning(f"Insufficient games in month {result_dict['month']}")
                continue
                
            # Round values
            result_dict['average_home_runs'] = round(result_dict['average_home_runs'], 2)
            result_dict['home_runs_per_game'] = round(result_dict['home_runs_per_game'], 2)
            
            processed_results.append(result_dict)

        return processed_results

    except (exceptions.Timeout, exceptions.GoogleAPIError) as e:
        logging.error(f"BigQuery error: {str(e)}")
        raise
    except Exception as e:
        logging.error(f"Error in analyze_monthly_home_runs: {str(e)}")
        raise


def analyze_home_scoring_vs_high_away_scores(min_away_score: int = 5) -> Dict[str, float]:
    """Analyzes home team scoring when away team scores above threshold.
    
    Args:
        min_away_score: Minimum away team score threshold (default 5)
    
    Returns:
        Dict: Home scoring statistics
        
    Raises:
        ValueError: If min_away_score is invalid
        BigQueryError: If there's an issue with the query execution
    """
    # Input validation
    if not isinstance(min_away_score, int):
        raise ValueError("min_away_score must be an integer")
        
    if min_away_score < 0:
        raise ValueError("min_away_score cannot be negative")
        
    if min_away_score > 30:  # Reasonable upper limit
        raise ValueError("min_away_score too high")

    try:
        query = """
        SELECT
            AVG(home_score) AS average_home_score,
            COUNT(*) as games_analyzed,
            MIN(home_score) as min_home_score,
            MAX(home_score) as max_home_score,
            AVG(away_score) as average_away_score
        FROM
            `mlb_data.combined_player_stats`
        WHERE
            away_score > @min_away_score
            AND home_score IS NOT NULL
            AND home_score >= 0
            AND away_score IS NOT NULL
        HAVING
            COUNT(*) >= 10  -- Minimum games threshold
        """

        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("min_away_score", "INT64", min_away_score)
            ]
        )

        query_job = bq_client.query(query, job_config=job_config)
        results = list(query_job.result(timeout=30))

        if not results:
            raise ValueError(f"No data found for away scores above {min_away_score}")

        result_dict = dict(results[0])
        
        # Validate results
        if result_dict['games_analyzed'] < 10:
            raise ValueError(f"Insufficient data: only {result_dict['games_analyzed']} games found")
            
        # Round values
        for key in ['average_home_score', 'average_away_score']:
            if key in result_dict:
                result_dict[key] = round(result_dict[key], 2)

        return result_dict

    except exceptions.Timeout as e:
        logging.error(f"Query timed out: {e}")
        raise
    except Exception as e:
        logging.error(f"Error in analyze_home_scoring_vs_high_away_scores: {e}")
        raise



def analyze_position_slugging(position: str, min_games: int = 100) -> Dict[str, Union[str, float, int]]:
    """Analyzes slugging percentage statistics for players in a specific position.
    
    Args:
        position: The player position to analyze
        min_games: Minimum games played threshold (default 100)
    
    Returns:
        Dict: Slugging percentage analysis for the position
    
    Raises:
        ValueError: If input parameters are invalid
        IndexError: If no data is found
        BigQueryError: If there's an issue with the BigQuery execution
        Exception: For other unexpected errors
    """
    # Valid MLB positions
    VALID_POSITIONS = {
        'P', 'C', '1B', '2B', '3B', 'SS', 'LF', 'CF', 'RF', 'DH', 
        'OF', 'IF', 'UT'  # Include utility and general positions
    }
    
    # Input validation
    if not isinstance(position, str):
        raise ValueError("Position must be a string")
        
    position = position.strip().upper()
    if not position:
        raise ValueError("Position cannot be empty")
        
    if position not in VALID_POSITIONS:
        raise ValueError(f"Invalid position. Must be one of: {', '.join(sorted(VALID_POSITIONS))}")
        
    if not isinstance(min_games, int):
        raise ValueError("min_games must be an integer")
        
    if min_games < 0:
        raise ValueError("min_games cannot be negative")
        
    if min_games > 162:  # Maximum games in an MLB regular season
        raise ValueError("min_games cannot exceed 162 (maximum games in MLB season)")

    try:
        query = """
        WITH PositionStats AS (
            SELECT
                position_name,
                player_id,
                slugging_percentage,
                games_played,
                at_bats
            FROM
                `mlb_data.combined_player_stats`
            WHERE
                position_name = @position
                AND games_played > @min_games
                AND slugging_percentage IS NOT NULL
                AND slugging_percentage >= 0
                AND slugging_percentage <= 4.0  -- Maximum theoretical slugging percentage
                AND at_bats >= 50  -- Minimum at-bats for meaningful stats
        )
        SELECT
            position_name,
            AVG(slugging_percentage) AS average_slugging_percentage,
            COUNT(DISTINCT player_id) as player_count,
            MIN(slugging_percentage) AS min_slugging,
            MAX(slugging_percentage) AS max_slugging,
            AVG(games_played) AS avg_games_played,
            COUNT(DISTINCT CASE WHEN slugging_percentage > 0.500 THEN player_id END) as players_above_500
        FROM
            PositionStats
        GROUP BY
            position_name
        HAVING
            COUNT(DISTINCT player_id) >= 5  -- Ensure sufficient sample size
        """

        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("position", "STRING", position),
                bigquery.ScalarQueryParameter("min_games", "INT64", min_games)
            ]
        )

        # Execute query with timeout
        query_job = bq_client.query(query, job_config=job_config)
        results = list(query_job.result(timeout=30))  # 30 second timeout
        
        if not results:
            logging.warning(f"No data found for position {position} with minimum {min_games} games")
            raise IndexError(f"No data found for position {position} with minimum {min_games} games")
            
        result_dict = dict(results[0])
        
        # Validate required fields
        required_fields = ['position_name', 'average_slugging_percentage', 
                         'player_count', 'min_slugging', 'max_slugging']
        if not all(field in result_dict for field in required_fields):
            logging.error("Missing required fields in query result")
            raise ValueError("Incomplete data returned from query")
            
        # Validate player count
        if result_dict['player_count'] < 5:
            logging.warning(f"Small sample size: only {result_dict['player_count']} qualified players")
            
        # Validate slugging percentages
        if not (0 <= result_dict['average_slugging_percentage'] <= 4.0):
            logging.error(f"Invalid average slugging percentage: {result_dict['average_slugging_percentage']}")
            raise ValueError("Invalid slugging percentage values")
            
        # Add derived statistics
        result_dict['slugging_range'] = round(result_dict['max_slugging'] - result_dict['min_slugging'], 3)
        result_dict['percentage_above_500'] = round(
            (result_dict['players_above_500'] / result_dict['player_count']) * 100, 1
        ) if result_dict['player_count'] > 0 else 0
        
        # Round numeric values for consistency
        result_dict['average_slugging_percentage'] = round(result_dict['average_slugging_percentage'], 3)
        result_dict['min_slugging'] = round(result_dict['min_slugging'], 3)
        result_dict['max_slugging'] = round(result_dict['max_slugging'], 3)
        result_dict['avg_games_played'] = round(result_dict['avg_games_played'], 1)
        
        return result_dict

    except IndexError as e:
        logging.error(f"No data found for position {position}")
        raise

    except exceptions.Timeout as e:
        logging.error(f"Query timed out: {e}")
        raise

    except exceptions.NotFound as e:
        logging.error(f"Table or dataset not found: {e}")
        raise

    except exceptions.BadRequest as e:
        logging.error(f"Invalid query or bad request: {e}")
        raise

    except exceptions.Forbidden as e:
        logging.error(f"Permission denied accessing BigQuery: {e}")
        raise

    except exceptions.GoogleAPIError as e:
        logging.error(f"BigQuery API error: {e}")
        raise

    except Exception as e:
        logging.error(f"Unexpected error in analyze_position_slugging: {e}")
        raise


def analyze_position_ops_percentile(min_games: int = 50) -> List[Dict[str, Union[str, float, int]]]:
    """Calculates the 90th percentile OPS for each position.
    
    Args:
        min_games: Minimum games played threshold (default 50)
    
    Returns:
        List[Dict]: 90th percentile OPS by position
    
    Raises:
        ValueError: If min_games parameter is invalid
        BigQueryError: If there's an issue with the BigQuery execution
        Exception: For other unexpected errors
    """
    # Input validation
    if not isinstance(min_games, int):
        raise ValueError("min_games must be an integer")
        
    if min_games < 0:
        raise ValueError("min_games cannot be negative")
        
    if min_games > 162:  # Maximum games in an MLB regular season
        raise ValueError("min_games cannot exceed 162 (maximum games in MLB season)")

    try:
        query = """
        SELECT
            position_name,
            APPROX_QUANTILES(on_base_plus_slugging, 100)[OFFSET(90)] AS ops_90th_percentile,
            COUNT(DISTINCT player_id) as qualified_players,
            MIN(on_base_plus_slugging) as min_ops,
            MAX(on_base_plus_slugging) as max_ops
        FROM
            `mlb_data.combined_player_stats`
        WHERE
            games_played >= @min_games
            AND position_name IS NOT NULL
            AND on_base_plus_slugging IS NOT NULL
            AND on_base_plus_slugging >= 0  -- OPS should not be negative
        GROUP BY
            position_name
        HAVING
            COUNT(DISTINCT player_id) >= 5  -- Ensure sufficient sample size
        ORDER BY
            ops_90th_percentile DESC
        """

        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("min_games", "INT64", min_games)
            ]
        )

        # Execute query with timeout
        query_job = bq_client.query(query, job_config=job_config)
        results = list(query_job.result(timeout=30))  # 30 second timeout
        
        if not results:
            logging.warning(f"No position data found meeting minimum {min_games} games threshold")
            return []
            
        # Validate and process results
        processed_results = []
        for row in results:
            result_dict = dict(row)
            
            # Validate required fields
            required_fields = ['position_name', 'ops_90th_percentile', 
                             'qualified_players', 'min_ops', 'max_ops']
            if not all(field in result_dict for field in required_fields):
                logging.error(f"Missing required fields in result for position {result_dict.get('position_name', 'unknown')}")
                continue
                
            # Validate OPS values
            if (result_dict['min_ops'] < 0 or 
                result_dict['max_ops'] < 0 or 
                result_dict['ops_90th_percentile'] < 0):
                logging.warning(f"Invalid OPS values detected for position {result_dict['position_name']}")
                continue
                
            # Validate qualified players count
            if result_dict['qualified_players'] < 5:
                logging.warning(f"Insufficient qualified players for position {result_dict['position_name']}")
                continue
                
            # Validate OPS relationships
            if (result_dict['min_ops'] > result_dict['max_ops'] or 
                result_dict['ops_90th_percentile'] < result_dict['min_ops'] or 
                result_dict['ops_90th_percentile'] > result_dict['max_ops']):
                logging.warning(f"Inconsistent OPS values for position {result_dict['position_name']}")
                continue
                
            processed_results.append(result_dict)
            
        return processed_results

    except exceptions.Timeout as e:
        logging.error(f"Query timed out: {e}")
        raise

    except exceptions.NotFound as e:
        logging.error(f"Table or dataset not found: {e}")
        raise

    except exceptions.BadRequest as e:
        logging.error(f"Invalid query or bad request: {e}")
        raise

    except exceptions.Forbidden as e:
        logging.error(f"Permission denied accessing BigQuery: {e}")
        raise

    except exceptions.GoogleAPIError as e:
        logging.error(f"BigQuery API error: {e}")
        raise

    except Exception as e:
        logging.error(f"Unexpected error in analyze_position_ops_percentile: {e}")
        raise

def analyze_homerun_win_correlation(start_year: int, end_year: int) -> List[Dict[str, Union[int, float]]]:
    """Analyzes correlation between team home runs and wins by season.
    
    Args:
        start_year: Starting season year
        end_year: Ending season year
    
    Returns:
        List[Dict]: Season-by-season correlation between home runs and wins
    
    Raises:
        ValueError: If year parameters are invalid
        BigQueryError: If there's an issue with the BigQuery execution
        Exception: For other unexpected errors
    """
    # Input validation
    current_year = datetime.now().year
    
    if not isinstance(start_year, int) or not isinstance(end_year, int):
        raise ValueError("Years must be integers")
        
    if start_year < 1876 or end_year > current_year:
        raise ValueError(f"Years must be between 1876 and {current_year}")
        
    if start_year > end_year:
        raise ValueError("Start year must be less than or equal to end year")
        
    if end_year - start_year > 100:
        raise ValueError("Date range cannot exceed 100 years")

    try:
        query = """
        WITH TeamWins AS (
            SELECT
                season,
                team_id,
                COUNTIF(home_score > away_score) AS wins
            FROM
                `mlb_data.combined_player_stats`
            WHERE
                season BETWEEN @start_year AND @end_year
                AND team_id IS NOT NULL
                AND home_score IS NOT NULL
                AND away_score IS NOT NULL
            GROUP BY 1, 2
            HAVING wins > 0  -- Ensure valid win counts
        ),
        TeamHomeruns AS (
            SELECT
                season,
                team_id,
                SUM(homeruns) AS total_homeruns
            FROM
                `mlb_data.combined_player_stats`
            WHERE
                season BETWEEN @start_year AND @end_year
                AND team_id IS NOT NULL
                AND homeruns IS NOT NULL
            GROUP BY 1, 2
            HAVING SUM(homeruns) >= 0  -- Ensure valid homerun counts
        )
        SELECT
            tw.season,
            CORR(th.total_homeruns, tw.wins) AS homerun_win_correlation
        FROM
            TeamWins AS tw
        JOIN
            TeamHomeruns AS th
        ON
            tw.season = th.season
            AND tw.team_id = th.team_id
        GROUP BY 1
        ORDER BY 1
        """

        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("start_year", "INT64", start_year),
                bigquery.ScalarQueryParameter("end_year", "INT64", end_year)
            ]
        )

        # Execute query with timeout
        query_job = bq_client.query(query, job_config=job_config)
        results = list(query_job.result(timeout=30))  # 30 second timeout
        
        if not results:
            logging.warning(f"No correlation data found for years {start_year}-{end_year}")
            return []
            
        # Validate and process results
        processed_results = []
        for row in results:
            result_dict = dict(row)
            
            # Validate required fields
            if not all(field in result_dict for field in ['season', 'homerun_win_correlation']):
                logging.error(f"Missing required fields in result for season {result_dict.get('season', 'unknown')}")
                continue
                
            # Validate correlation value
            correlation = result_dict['homerun_win_correlation']
            if correlation is not None and (correlation < -1 or correlation > 1):
                logging.warning(f"Invalid correlation value {correlation} for season {result_dict['season']}")
                continue
                
            # Validate season
            if not (start_year <= result_dict['season'] <= end_year):
                logging.warning(f"Season {result_dict['season']} outside requested range")
                continue
                
            processed_results.append(result_dict)
            
        return processed_results

    except exceptions.Timeout as e:
        logging.error(f"Query timed out: {e}")
        raise

    except exceptions.NotFound as e:
        logging.error(f"Table or dataset not found: {e}")
        raise

    except exceptions.BadRequest as e:
        logging.error(f"Invalid query or bad request: {e}")
        raise

    except exceptions.Forbidden as e:
        logging.error(f"Permission denied accessing BigQuery: {e}")
        raise

    except exceptions.GoogleAPIError as e:
        logging.error(f"BigQuery API error: {e}")
        raise

    except Exception as e:
        logging.error(f"Unexpected error in analyze_homerun_win_correlation: {e}")
        raise


def analyze_position_batting_efficiency(min_at_bats: int = 200) -> List[Dict[str, Union[str, int, float]]]:
    """Calculates batting efficiency metrics by position.
    
    Args:
        min_at_bats: Minimum at-bats threshold (default 200)
    
    Returns:
        List[Dict]: Batting efficiency statistics by position
    
    Raises:
        ValueError: If min_at_bats parameter is invalid
        BigQueryError: If there's an issue with the BigQuery execution
        Exception: For other unexpected errors
    """
    # Input validation
    if not isinstance(min_at_bats, int):
        raise ValueError("min_at_bats must be an integer")
        
    if min_at_bats < 0:
        raise ValueError("min_at_bats cannot be negative")
        
    if min_at_bats > 700:  # Maximum realistic at-bats in a season
        raise ValueError("min_at_bats cannot exceed 700 (maximum realistic season at-bats)")

    try:
        query = """
        SELECT
            position_name,
            SUM(at_bats) as total_at_bats,
            SUM(hits) as total_hits,
            SAFE_DIVIDE(SUM(at_bats), NULLIF(SUM(hits), 0)) AS avg_at_bats_per_hit,
            COUNT(DISTINCT player_id) as qualified_players,
            AVG(batting_average) as avg_batting_average
        FROM
            `mlb_data.combined_player_stats`
        WHERE
            at_bats >= @min_at_bats
            AND position_name IS NOT NULL
            AND hits IS NOT NULL
            AND hits >= 0
            AND at_bats > 0
        GROUP BY
            position_name
        HAVING
            COUNT(DISTINCT player_id) >= 5  -- Ensure sufficient sample size
            AND SUM(hits) > 0  -- Prevent division by zero
        ORDER BY
            avg_at_bats_per_hit ASC
        """

        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("min_at_bats", "INT64", min_at_bats)
            ]
        )

        # Execute query with timeout
        query_job = bq_client.query(query, job_config=job_config)
        results = list(query_job.result(timeout=30))  # 30 second timeout
        
        if not results:
            logging.warning(f"No position data found meeting minimum {min_at_bats} at-bats threshold")
            return []
            
        # Validate and process results
        processed_results = []
        for row in results:
            result_dict = dict(row)
            
            try:
                # Validate required fields
                required_fields = ['position_name', 'total_at_bats', 'total_hits', 
                                 'avg_at_bats_per_hit', 'qualified_players', 
                                 'avg_batting_average']
                if not all(field in result_dict for field in required_fields):
                    logging.error(f"Missing required fields in result for position {result_dict.get('position_name', 'unknown')}")
                    continue
                    
                # Validate numeric values
                if result_dict['total_at_bats'] <= 0:
                    logging.warning(f"Invalid total_at_bats for position {result_dict['position_name']}")
                    continue
                    
                if result_dict['total_hits'] < 0:
                    logging.warning(f"Invalid total_hits for position {result_dict['position_name']}")
                    continue
                    
                if result_dict['avg_at_bats_per_hit'] <= 0:
                    logging.warning(f"Invalid avg_at_bats_per_hit for position {result_dict['position_name']}")
                    continue
                    
                # Validate batting average
                if not (0 <= result_dict['avg_batting_average'] <= 1):
                    logging.warning(f"Invalid batting average for position {result_dict['position_name']}")
                    continue
                    
                # Validate qualified players count
                if result_dict['qualified_players'] < 5:
                    logging.warning(f"Insufficient qualified players for position {result_dict['position_name']}")
                    continue
                    
                # Validate relationship between hits and at-bats
                if result_dict['total_hits'] > result_dict['total_at_bats']:
                    logging.warning(f"More hits than at-bats for position {result_dict['position_name']}")
                    continue
                
                # Round floating point values for consistency
                result_dict['avg_at_bats_per_hit'] = round(result_dict['avg_at_bats_per_hit'], 3)
                result_dict['avg_batting_average'] = round(result_dict['avg_batting_average'], 3)
                
                processed_results.append(result_dict)
                
            except (TypeError, ValueError) as e:
                logging.error(f"Error processing result for position {result_dict.get('position_name', 'unknown')}: {e}")
                continue
            
        return processed_results

    except exceptions.Timeout as e:
        logging.error(f"Query timed out: {e}")
        raise

    except exceptions.NotFound as e:
        logging.error(f"Table or dataset not found: {e}")
        raise

    except exceptions.BadRequest as e:
        logging.error(f"Invalid query or bad request: {e}")
        raise

    except exceptions.Forbidden as e:
        logging.error(f"Permission denied accessing BigQuery: {e}")
        raise

    except exceptions.GoogleAPIError as e:
        logging.error(f"BigQuery API error: {e}")
        raise

    except Exception as e:
        logging.error(f"Unexpected error in analyze_position_batting_efficiency: {e}")
        raise

def analyze_team_weight_trends(season: int) -> List[Dict[str, Union[str, int, float]]]:
    """Identifies teams with highest average player weights and related stats.
    
    Args:
        season: The season year to analyze
    
    Returns:
        List[Dict]: Teams ranked by average player weight with additional metrics
    
    Raises:
        ValueError: If season is invalid
        BigQueryError: If there's an issue with the BigQuery execution
        Exception: For other unexpected errors
    """
    # Input validation
    current_year = datetime.now().year
    
    if not isinstance(season, int):
        raise ValueError("Season must be an integer")
    
    if season < 1876 or season > current_year:
        raise ValueError(f"Season must be between 1876 and {current_year}")

    try:
        query = """
        SELECT
            season,
            team_name,
            AVG(weight) AS average_player_weight,
            MIN(weight) AS min_weight,
            MAX(weight) AS max_weight,
            COUNT(DISTINCT player_id) AS roster_size
        FROM
            `mlb_data.combined_player_stats`
        WHERE
            season = @season
            AND weight IS NOT NULL  -- Ensure weight values exist
            AND weight > 0  -- Basic validation for weight
        GROUP BY
            season,
            team_name
        HAVING
            COUNT(DISTINCT player_id) > 0  -- Ensure we have valid players
        ORDER BY
            average_player_weight DESC
        """

        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("season", "INT64", season)
            ]
        )

        # Execute query with timeout
        query_job = bq_client.query(query, job_config=job_config)
        
        # Wait for query to complete and catch any execution errors
        results = list(query_job.result(timeout=30))  # 30 second timeout
        
        if not results:
            logging.warning(f"No team weight data found for season {season}")
            return []
            
        # Validate and process results
        processed_results = []
        for row in results:
            result_dict = dict(row)
            
            # Validate required fields
            required_fields = ['season', 'team_name', 'average_player_weight', 
                             'min_weight', 'max_weight', 'roster_size']
            if not all(field in result_dict for field in required_fields):
                logging.error(f"Missing required fields in result for team {result_dict.get('team_name', 'unknown')}")
                continue
            
            # Validate weight values
            if (result_dict['min_weight'] <= 0 or 
                result_dict['max_weight'] <= 0 or 
                result_dict['average_player_weight'] <= 0):
                logging.warning(f"Invalid weight values detected for team {result_dict['team_name']}")
                continue
                
            # Validate roster size
            if result_dict['roster_size'] <= 0:
                logging.warning(f"Invalid roster size for team {result_dict['team_name']}")
                continue
                
            # Basic sanity check for weight values
            if (result_dict['min_weight'] > result_dict['max_weight'] or 
                result_dict['average_player_weight'] < result_dict['min_weight'] or 
                result_dict['average_player_weight'] > result_dict['max_weight']):
                logging.warning(f"Inconsistent weight values for team {result_dict['team_name']}")
                continue
                
            processed_results.append(result_dict)
            
        return processed_results

    except exceptions.Timeout as e:
        logging.error(f"Query timed out: {e}")
        raise

    except exceptions.NotFound as e:
        logging.error(f"Table or dataset not found: {e}")
        raise

    except exceptions.BadRequest as e:
        logging.error(f"Invalid query or bad request: {e}")
        raise

    except exceptions.Forbidden as e:
        logging.error(f"Permission denied accessing BigQuery: {e}")
        raise

    except exceptions.GoogleAPIError as e:
        logging.error(f"BigQuery API error: {e}")
        raise

    except Exception as e:
        logging.error(f"Unexpected error in analyze_team_weight_trends: {e}")
        raise

def analyze_stolen_base_efficiency(season: int) -> List[Dict[str, Union[str, int, float]]]:
    """Analyzes team's stolen base efficiency in winning home games.
    
    Args:
        season: The season year to analyze
    
    Returns:
        List[Dict]: Teams ranked by stolen base efficiency metrics
    
    Raises:
        ValueError: If season is invalid
        BigQueryError: If there's an issue with the BigQuery execution
        Exception: For other unexpected errors
    """
    # Input validation
    current_year = datetime.now().year
    
    if not isinstance(season, int):
        raise ValueError("Season must be an integer")
    
    if season < 1876 or season > current_year:  # MLB's first season was 1876
        raise ValueError(f"Season must be between 1876 and {current_year}")

    try:
        query = """
        SELECT
            season,
            team_name,
            AVG(CAST(stolen_bases AS FLOAT64)) / COUNT(DISTINCT game_pk) 
                AS avg_stolen_bases_per_game,
            SUM(stolen_bases) as total_stolen_bases,
            COUNT(DISTINCT game_pk) as games_played
        FROM
            `mlb_data.combined_player_stats`
        WHERE
            season = @season
            AND home_score > away_score
        GROUP BY
            season,
            team_name
        ORDER BY
            avg_stolen_bases_per_game DESC
        """

        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("season", "INT64", season)
            ]
        )

        # Execute query with timeout
        query_job = bq_client.query(query, job_config=job_config)
        
        # Wait for query to complete and catch any execution errors
        results = list(query_job.result(timeout=30))  # 30 second timeout
        
        if not results:
            logging.warning(f"No data found for season {season}")
            return []
            
        # Validate results
        processed_results = []
        for row in results:
            result_dict = dict(row)
            
            # Ensure required fields are present
            required_fields = ['season', 'team_name', 'avg_stolen_bases_per_game', 
                             'total_stolen_bases', 'games_played']
            if not all(field in result_dict for field in required_fields):
                logging.error(f"Missing required fields in result for team {result_dict.get('team_name', 'unknown')}")
                continue
                
            # Validate numeric fields
            if result_dict['games_played'] <= 0:
                logging.warning(f"Invalid games_played value for team {result_dict['team_name']}")
                continue
                
            processed_results.append(result_dict)
            
        return processed_results

    except exceptions.Timeout as e:
        logging.error(f"Query timed out: {e}")
        raise

    except exceptions.NotFound as e:
        logging.error(f"Table or dataset not found: {e}")
        raise

    except exceptions.BadRequest as e:
        logging.error(f"Invalid query or bad request: {e}")
        raise

    except exceptions.Forbidden as e:
        logging.error(f"Permission denied accessing BigQuery: {e}")
        raise

    except exceptions.GoogleAPIError as e:
        logging.error(f"BigQuery API error: {e}")
        raise

    except Exception as e:
        logging.error(f"Unexpected error in analyze_stolen_base_efficiency: {e}")
        raise


def analyze_ops_percentile_trends() -> list:
    """Calculates 90th percentile OPS trends across seasons.
    
    Returns:
        list: Season-by-season 90th percentile OPS values
        
    Raises:
        BigQueryError: If there's an issue with the BigQuery execution
        Exception: For other unexpected errors
    """
    try:
        query = """
        SELECT
            season,
            APPROX_QUANTILES(on_base_plus_slugging, 100)[OFFSET(90)] AS ops_90th_percentile,
            AVG(on_base_plus_slugging) AS avg_ops,
            COUNT(DISTINCT player_id) as qualified_players
        FROM
            `mlb_data.combined_player_stats`
        GROUP BY
            season
        ORDER BY
            season DESC
        """
        query_job = bq_client.query(query)
        
        # Check if the query job completed successfully
        query_job.result()  # This will raise an exception if the query failed
        
        results = [dict(row) for row in query_job]
        
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
        logging.error(f"Unexpected error in analyze_ops_percentile_trends: {e}")
        raise



def analyze_near_cycle_players(season: int, team_name: str, last_n_games: int = 5) -> List[Dict[str, Union[str, int]]]:
    """Finds players who nearly hit for the cycle in team's last N games of a season.
    
    Args:
        season: The season to analyze
        team_name: The team name to analyze
        last_n_games: Number of last games to analyze (default 5)
        
    Returns:
        List[Dict]: Players who had 3+ hits including at least one single, double, triple, and homer
        
    Raises:
        ValueError: If parameters are invalid
        BigQueryError: If there's an issue with the BigQuery execution
    """
    if not isinstance(season, int) or not isinstance(team_name, str):
        raise ValueError("Season must be an integer and team_name must be a string")
        
    try:
        query = """
        WITH LastFiveGames AS (
            SELECT DISTINCT
                game_date
            FROM
                `mlb_data.combined_player_stats`
            WHERE
                season = @season
                AND team_name = @team_name
            ORDER BY game_date DESC
            LIMIT @last_n_games
        )
        SELECT DISTINCT
            first_name,
            last_name
        FROM
            `mlb_data.combined_player_stats`
        WHERE
            season = @season
            AND team_name = @team_name
            AND game_date IN (SELECT game_date FROM LastFiveGames)
            AND hits >= 3
            AND doubles >= 1
            AND triples >= 1
            AND homeruns >= 1
        ORDER BY last_name
        """
        
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("season", "INT64", season),
                bigquery.ScalarQueryParameter("team_name", "STRING", team_name),
                bigquery.ScalarQueryParameter("last_n_games", "INT64", last_n_games)
            ]
        )
        
        query_job = bq_client.query(query, job_config=job_config)
        results = list(query_job.result(timeout=30))
        
        if not results:
            logging.warning(f"No near-cycle performances found for {team_name} in last {last_n_games} games of {season}")
            return []
            
        return [dict(row) for row in results]
        
    except Exception as e:
        logging.error(f"Error in analyze_near_cycle_players: {e}")
        raise


def get_player_recent_streaks(player_name: str, stat: str, last_n_games: int = 10, threshold: float = 0.300):
    """
    Finds recent games where a player exceeded a threshold for a specific statistic.

    Args:
        player_name: The full name of the player.
        stat: The statistic to track (e.g., 'batting_average', 'on_base_plus_slugging', 'homeruns').
        last_n_games: The number of recent games to consider (default: 10).
        threshold: The threshold value for the statistic (default: 0.300).

    Returns:
        list: A list of dictionaries containing game_date and the specified stat
              for games where the threshold was met.
    """
    try:
        query = f"""
        SELECT
            game_date,
            {stat}
        FROM
            `mlb_data.combined_player_stats`
        WHERE
            full_name = @player_name
        ORDER BY
            game_date DESC
        LIMIT @last_n_games
        """

        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("player_name", "STRING", player_name),
                bigquery.ScalarQueryParameter("last_n_games", "INT64", last_n_games)
            ]
        )

        query_job = bq_client.query(query, job_config=job_config)
        results = list(query_job.result())

        # Changed from row.to_dict() to dict(row)
        streaks = [dict(row) for row in results if row.get(stat) is not None and row[stat] >= threshold]

        return streaks

    except Exception as e:
        logging.error(f"Error in get_player_recent_streaks: {e}")
        return []


def get_team_clutch_moments(team_name: str, season: int, min_run_difference: int = 2):
    """
    Identifies games where a team had a significant comeback or held a tight lead.

    Args:
        team_name: The name of the team.
        season: The season to analyze.
        min_run_difference: The minimum run difference to consider a "clutch" moment.

    Returns:
        list: A list of dictionaries describing the clutch moments.
    """
    try:
        query = """
        SELECT
            game_date,
            home_team.team_name AS home_team,
            away_team.team_name AS away_team,
            home_score,
            away_score,
            CASE
                WHEN home_team.team_name = @team_name AND home_score > away_score AND ABS(home_score - away_score) <= @min_run_difference THEN 'Tight Win'
                WHEN away_team.team_name = @team_name AND away_score > home_score AND ABS(away_score - home_score) <= @min_run_difference THEN 'Tight Win'
                WHEN home_team.team_name = @team_name AND home_score > away_score AND ABS(original_home_score - original_away_score) > @min_run_difference THEN 'Comeback Win'
                WHEN away_team.team_name = @team_name AND away_score > home_score AND ABS(original_away_score - original_home_score) > @min_run_difference THEN 'Comeback Win'
                ELSE NULL
            END AS clutch_moment_type
        FROM
            `mlb_data.combined_game_stats`
        JOIN
            `mlb_data.teams` AS home_team ON combined_game_stats.home_team_id = home_team.team_id
        JOIN
            `mlb_data.teams` AS away_team ON combined_game_stats.away_team_id = away_team.team_id
        WHERE
            (home_team.team_name = @team_name OR away_team.team_name = @team_name)
            AND season = @season
            AND clutch_moment_type IS NOT NULL
        ORDER BY
            game_date DESC
        """

        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("team_name", "STRING", team_name),
                bigquery.ScalarQueryParameter("season", "INT64", season),
                bigquery.ScalarQueryParameter("min_run_difference", "INT64", min_run_difference)
            ]
        )
        query_job = bq_client.query(query, job_config=job_config)
        results = list(query_job.result())
        return [dict(row) for row in results]
    except Exception as e:
        logging.error(f"Error in get_team_clutch_moments: {e}")
        return []


def analyze_team_bunt_tendency(team_name: str, season: int, situation: str = 'RISP'):
    """
    Analyzes how often a team attempts to bunt in specific situations.

    Args:
        team_name: The name of the team.
        season: The season to analyze.
        situation: The specific game situation to analyze (e.g., 'RISP' - Runners in Scoring Position, 'LeadOff').

    Returns:
        list: Statistics on the team's bunt tendencies in the specified situation.
    """
    try:
        query = f"""
        SELECT
            COUNT(*) AS total_opportunities,
            SUM(CASE WHEN rbi = 0 AND hits = 0 THEN 1 ELSE 0 END) AS bunt_attempts,
            AVG(CASE WHEN rbi = 0 AND hits = 0 THEN 1 ELSE 0 END) AS bunt_rate
        FROM
            `mlb_data.combined_player_stats`
        WHERE
            team_name = @team_name
            AND season = @season
            {'AND runs_batted_in > 0' if situation == 'RISP' else ''}
        """

        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("team_name", "STRING", team_name),
                bigquery.ScalarQueryParameter("season", "INT64", season)
            ]
        )
        query_job = bq_client.query(query, job_config=job_config)
        results = list(query_job.result())
        return [dict(row) for row in results]
    except Exception as e:
        logging.error(f"Error in analyze_team_bunt_tendency: {e}")
        return []


def compare_rookie_season_to_veteran(rookie_name: str, veteran_name: str, stat_categories: str = ''):
    """
    Compares a rookie's stats to a veteran player's stats in their early career.

    Args:
        rookie_name: The full name of the rookie player.
        veteran_name: The full name of the veteran player.
        stat_categories: Comma-separated string of statistics to compare (e.g., 'batting_average,homeruns').
                        If empty, uses default stats.

    Returns:
        list: A comparison of the specified stats for both players.
    """
    default_stats = ['batting_average', 'on_base_percentage', 'slugging_percentage', 
                    'on_base_plus_slugging', 'homeruns', 'rbi', 'stolen_bases']
    
    # Parse stat_categories or use defaults
    stats_to_use = stat_categories.split(',') if stat_categories else default_stats
    stats_to_use = [stat.strip() for stat in stats_to_use]  # Clean up any whitespace

    try:
        query = f"""
        WITH RookieStats AS (
            SELECT
                season,
                {','.join(stats_to_use)}
            FROM
                `mlb_data.combined_player_stats`
            WHERE
                full_name = @rookie_name
            ORDER BY season
            LIMIT 1
        ),
        VeteranStats AS (
            SELECT
                season,
                {','.join(stats_to_use)}
            FROM
                `mlb_data.combined_player_stats`
            WHERE
                full_name = @veteran_name
            ORDER BY season
            LIMIT 1
        )
        SELECT
            'Rookie' AS player_type,
            RookieStats.*
        FROM RookieStats
        UNION ALL
        SELECT
            'Veteran' AS player_type,
            VeteranStats.*
        FROM VeteranStats
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("rookie_name", "STRING", rookie_name),
                bigquery.ScalarQueryParameter("veteran_name", "STRING", veteran_name)
            ]
        )
        query_job = bq_client.query(query, job_config=job_config)
        results = list(query_job.result())
        return [dict(row) for row in results]
    except Exception as e:
        logging.error(f"Error in compare_rookie_season_to_veteran: {e}")
        return []

def identify_undervalued_players(season: int, min_games_played: int = 50, ops_threshold: float = 0.800):
    """
    Identifies players with high offensive output (OPS) who might be playing on lower-performing teams.

    Args:
        season: The season to analyze.
        min_games_played: Minimum number of games played to be considered.
        ops_threshold: The minimum OPS to be considered high-performing.

    Returns:
        list: Players who meet the criteria.
    """
    try:
        query = """
        SELECT
            p.full_name,
            p.team_name,
            p.on_base_plus_slugging,
            t.wins,
            t.losses
        FROM
            `mlb_data.combined_player_stats` p
        JOIN
            `mlb_data.standings` t ON p.team_id = t.team_id AND p.season = t.season
        WHERE
            p.season = @season
            AND p.games_played >= @min_games_played
            AND p.on_base_plus_slugging >= @ops_threshold
        ORDER BY
            t.wins ASC
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("season", "INT64", season),
                bigquery.ScalarQueryParameter("min_games_played", "INT64", min_games_played),
                bigquery.ScalarQueryParameter("ops_threshold", "FLOAT64", ops_threshold)
            ]
        )
        query_job = bq_client.query(query, job_config=job_config)
        results = list(query_job.result())
        return [dict(row) for row in results]
    except Exception as e:
        logging.error(f"Error in identify_undervalued_players: {e}")
        return []


def predict_matchup_outcome_by_stats(home_team_name: str, away_team_name: str, season: int, stat: str = 'on_base_plus_slugging'):
    """
    Predicts the outcome of a matchup based on a comparison of team statistics.

    Args:
        home_team_name: The name of the home team.
        away_team_name: The name of the away team.
        season: The season to analyze.
        stat: The statistic to use for comparison (e.g., 'on_base_plus_slugging', 'batting_average').

    Returns:
        list: A prediction of the likely winner based on the chosen statistic.
    """
    try:
        query = f"""
        WITH HomeTeamStats AS (
            SELECT
                AVG({stat}) AS home_team_avg_stat
            FROM
                `mlb_data.combined_player_stats`
            WHERE
                team_name = @home_team_name
                AND season = @season
        ),
        AwayTeamStats AS (
            SELECT
                AVG({stat}) AS away_team_avg_stat
            FROM
                `mlb_data.combined_player_stats`
            WHERE
                team_name = @away_team_name
                AND season = @season
        )
        SELECT
            (SELECT home_team_avg_stat FROM HomeTeamStats) AS home_team_avg_stat,
            (SELECT away_team_avg_stat FROM AwayTeamStats) AS away_team_avg_stat,
            CASE
                WHEN (SELECT home_team_avg_stat FROM HomeTeamStats) > (SELECT away_team_avg_stat FROM AwayTeamStats) THEN @home_team_name
                WHEN (SELECT away_team_avg_stat FROM AwayTeamStats) > (SELECT home_team_avg_stat FROM HomeTeamStats) THEN @away_team_name
                ELSE 'Tie/Too Close to Call'
            END AS predicted_winner
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("home_team_name", "STRING", home_team_name),
                bigquery.ScalarQueryParameter("away_team_name", "STRING", away_team_name),
                bigquery.ScalarQueryParameter("season", "INT64", season)
            ]
        )
        query_job = bq_client.query(query, job_config=job_config)
        results = list(query_job.result())
        return [dict(row) for row in results]
    except Exception as e:
        logging.error(f"Error in predict_matchup_outcome_by_stats: {e}")
        return []


def get_player_stat_ranks_vs_peers(player_name: str, season: int, stat: str, position: str = None) -> List[Dict]:
    """
    Gets a player's rank in a specific statistic compared to others at the same position (optional) or all players.

    Args:
        player_name: The full name of the player.
        season: The season to analyze.
        stat: The statistic to rank by (e.g., 'homeruns', 'rbi', 'batting_average').
        position: (Optional) The player's position to filter peers (e.g., 'OF', 'SS'). If None, compare against all players.

    Returns:
        List[Dict]: A list containing a dictionary with the player's rank and total count of players considered.
    """
    try:
        query = f"""
        WITH PlayerStats AS (
            SELECT
                full_name,
                {stat},
                RANK() OVER (ORDER BY {stat} DESC) as overall_rank
            FROM
                `mlb_data.combined_player_stats`
            WHERE
                season = @season
        ),
        PositionStats AS (
            SELECT
                full_name,
                {stat},
                RANK() OVER (ORDER BY {stat} DESC) as position_rank
            FROM
                `mlb_data.combined_player_stats`
            WHERE
                season = @season
                {'AND position_code = @position' if position else ''}
        )
        SELECT
            (SELECT {stat} FROM PlayerStats WHERE full_name = @player_name) as player_stat,
            (SELECT overall_rank FROM PlayerStats WHERE full_name = @player_name) as overall_rank,
            (SELECT COUNT(*) FROM PlayerStats) as total_players{f""",
            (SELECT position_rank FROM PositionStats WHERE full_name = @player_name) as position_rank,
            (SELECT COUNT(*) FROM PositionStats) as total_position_players""" if position else ''}
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("player_name", "STRING", player_name),
                bigquery.ScalarQueryParameter("season", "INT64", season),
                bigquery.ScalarQueryParameter("position", "STRING", position) if position else None
            ]
        )
        query_job = bq_client.query(query, job_config=job_config)
        results = list(query_job.result())
        return [dict(row) for row in results]
    except Exception as e:
        logging.error(f"Error in get_player_stat_ranks_vs_peers: {e}")
        return []

def generate_mlb_analysis(contents: str) -> str:
    """
    Generates MLB analysis using Gemini with specified tools.

    Args:
        contents: The prompt or question for the analysis.

    Returns:
        The text response from the Gemini model.  Returns an empty string if there's an error.
    """
    client = genai.Client(vertexai=True, project="gem-creation", location="us-central1")  # Initialize client only once
    MODEL_ID = "gemini-2.0-flash-exp"  # Define Model ID only once

    try:
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=contents,
            config=GenerateContentConfig(
                tools=[
                    get_player_highest_ops,
                    analyze_player_performance,
                    calculate_team_home_advantage,
                    analyze_position_slugging,
                    analyze_team_strikeouts,
                    analyze_monthly_home_runs,
                    analyze_weight_performance,
                    analyze_homerun_win_correlation,
                    analyze_position_ops_percentile,
                    analyze_position_batting_efficiency,
                    analyze_team_weight_trends,
                    analyze_stolen_base_efficiency,
                    analyze_ops_percentile_trends,
                    analyze_home_scoring_vs_high_away_scores,
                    analyze_near_cycle_players,
                    get_player_recent_streaks,
                    get_team_clutch_moments,
                    analyze_team_bunt_tendency,
                    compare_rookie_season_to_veteran,
                    identify_undervalued_players,
                    predict_matchup_outcome_by_stats,
                ],
                temperature=0,  # Ensure deterministic output for consistent results
            ),
        )
        return response.text
    except Exception as e:
        print(f"An error occurred: {e}")
        return ""