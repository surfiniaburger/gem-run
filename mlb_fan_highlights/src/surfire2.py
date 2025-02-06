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

def get_player_highest_ops(year: int) -> Dict[str, Union[str, float]]:
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
    
    if not isinstance(year, int):
        raise ValueError("Season must be an integer")
        
    if year < 1876 or year > current_year:
        raise ValueError(f"Season must be between 1876 and {current_year}")

    try:
        query = """
        SELECT
            first_name,
            last_name,
            on_base_plus_slg,
            ab
        FROM
            `mlb_data.player_stats`
        WHERE
            season = @year
            AND on_base_plus_slg IS NOT NULL
            AND on_base_plus_slug >= 0
            AND ab >= 30      -- Minimum at-bats threshold
        QUALIFY
            ROW_NUMBER() OVER (PARTITION BY season 
                             ORDER BY on_base_plus_slg DESC) = 1
        """

        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("year", "INTEGER", year)
            ]
        )

        # Execute query with timeout
        query_job = bq_client.query(query, job_config=job_config)
        results = list(query_job.result(timeout=30))  # 30 second timeout
        
        if not results:
            logging.warning(f"No qualified players found for season {year}")
            raise IndexError(f"No qualified players found for season {year}")
            
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
        logging.error(f"No data found for season {year}")
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
            APPROX_QUANTILES(on_base_plus_slg, 100)[OFFSET(90)] AS ops_90th_percentile,
            AVG(on_base_plus_slg) AS avg_ops,
            COUNT(DISTINCT player_id) as qualified_players
        FROM
            `mlb_data.player_stats`
        GROUP BY
            year
        ORDER BY
            year DESC
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


def fetch_home_team_performance(limit: int = 1000) -> list:
    """
    Fetches the home team performance data, calculating wins and losses for each team.
    
    Args:
        limit (int): Maximum number of results to return. Must be between 1 and 1000.
    
    Returns:
        list: A list of dictionaries containing team names, home wins, and home losses.
    
    Raises:
        ValueError: If the limit parameter is invalid.
        BigQueryError: If there's an issue with the BigQuery execution.
        Exception: For other unexpected errors.
    """
    # Input validation
    if not isinstance(limit, int):
        raise ValueError("Limit must be an integer")
    if limit < 1 or limit > 1000:
        raise ValueError("Limit must be between 1 and 1000")

    try:
        # Define the query with parameterized limit
        query = """
        SELECT 
          home_team_name AS team_name,
          CAST(COUNT(CASE WHEN home_score > away_score THEN 1 END) AS INT64) AS home_wins,
          CAST(COUNT(CASE WHEN away_score > home_score THEN 1 END) AS INT64) AS home_losses
        FROM 
          `gem-rush-007.mlb_data_2024.games`
        WHERE 
          status = 'Final'
          AND home_team_name IS NOT NULL
          AND home_score IS NOT NULL 
          AND away_score IS NOT NULL
        GROUP BY 
          home_team_name
        ORDER BY 
          home_wins DESC
        LIMIT @limit
        """
        
        # Define the query parameters
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("limit", "INT64", limit),
            ]
        )
        
        # Initialize BigQuery client
        bq_client = bigquery.Client()
        
        # Execute the query with parameters
        query_job = bq_client.query(query, job_config=job_config)
        
        # Ensure the query completes successfully
        query_job.result()
        
        # Fetch and process the results with validation
        results = []
        for row in query_job:
            row_dict = dict(row)
            
            # Validate team name
            if not row_dict.get('team_name'):
                logging.warning("Skipping record with missing team name")
                continue
                
            # Validate numeric fields
            for field in ['home_wins', 'home_losses']:
                if field in row_dict:
                    try:
                        row_dict[field] = int(row_dict[field])
                    except (TypeError, ValueError):
                        logging.warning(f"Invalid {field} value for {row_dict['team_name']}: {row_dict[field]}")
                        row_dict[field] = 0
                else:
                    row_dict[field] = 0
            
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
        logging.error(f"Unexpected error in fetch_home_team_performance: {e}")
        raise


def fetch_team_performance_by_venue(limit: int = 1000) -> list:
    """
    Fetches team performance at different venues, including games played and wins at each venue.
    
    Args:
        limit (int): Maximum number of results to return. Must be between 1 and 1000.
    
    Returns:
        list: A list of dictionaries containing venue names, team names, 
              games played, and wins at the venue.
    
    Raises:
        ValueError: If the limit parameter is invalid.
        BigQueryError: If there's an issue with the BigQuery execution.
        Exception: For other unexpected errors.
    """
    # Input validation
    if not isinstance(limit, int):
        raise ValueError("Limit must be an integer")
    if limit < 1 or limit > 1000:
        raise ValueError("Limit must be between 1 and 1000")

    try:
        # Define the query with parameterized limit
        query = """
        SELECT 
          venue_name,
          home_team_name AS team_name,
          CAST(COUNT(*) AS INT64) AS games_played,
          CAST(COUNT(CASE WHEN home_score > away_score THEN 1 END) AS INT64) AS wins_at_venue
        FROM 
          `gem-rush-007.mlb_data_2024.games`
        WHERE 
          status = 'Final'
          AND venue_name IS NOT NULL
          AND home_team_name IS NOT NULL
          AND home_score IS NOT NULL
          AND away_score IS NOT NULL
        GROUP BY 
          venue_name, home_team_name
        HAVING 
          games_played > 0
        ORDER BY 
          wins_at_venue DESC
        LIMIT @limit
        """
        
        # Define the query parameters
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("limit", "INT64", limit),
            ]
        )
        
        # Initialize BigQuery client
        bq_client = bigquery.Client()
        
        # Execute the query with parameters
        query_job = bq_client.query(query, job_config=job_config)
        
        # Ensure the query completes successfully
        query_job.result()
        
        # Fetch and process the results with validation
        results = []
        for row in query_job:
            row_dict = dict(row)
            
            # Validate required fields
            if not row_dict.get('venue_name') or not row_dict.get('team_name'):
                logging.warning("Skipping record with missing venue or team name")
                continue
            
            # Validate and convert numeric fields
            try:
                row_dict['games_played'] = int(row_dict.get('games_played', 0))
                row_dict['wins_at_venue'] = int(row_dict.get('wins_at_venue', 0))
                
                # Additional validation
                if row_dict['wins_at_venue'] > row_dict['games_played']:
                    logging.warning(f"Invalid data: wins ({row_dict['wins_at_venue']}) > games played ({row_dict['games_played']}) for {row_dict['team_name']} at {row_dict['venue_name']}")
                    continue
                
            except (TypeError, ValueError) as e:
                logging.warning(f"Invalid numeric data for {row_dict.get('team_name')} at {row_dict.get('venue_name')}: {e}")
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
        logging.error(f"Unexpected error in fetch_team_performance_by_venue: {e}")
        raise

def fetch_upcoming_games(limit: int = 10) -> list:
    """
    Fetches upcoming scheduled MLB games with their date, teams, and status.
    
    Args:
        limit (int): Maximum number of results to return. Must be between 1 and 100.
    
    Returns:
        list: A list of dictionaries containing game date, home team, 
              away team, and game status for upcoming games.
    
    Raises:
        ValueError: If the limit parameter is invalid.
        BigQueryError: If there's an issue with the BigQuery execution.
        Exception: For other unexpected errors.
    """
    # Input validation
    if not isinstance(limit, int):
        raise ValueError("Limit must be an integer")
    if limit < 1 or limit > 100:
        raise ValueError("Limit must be between 1 and 100")

    try:
        # Define the query with parameterized limit
        query = """
        SELECT 
          game_date,
          home_team_name,
          away_team_name,
          status
        FROM 
          `gem-rush-007.mlb_data_2024.games`
        WHERE 
          game_date > CURRENT_DATE() 
          AND status = 'Scheduled'
          AND home_team_name IS NOT NULL
          AND away_team_name IS NOT NULL
          AND game_date IS NOT NULL
        ORDER BY 
          game_date ASC
        LIMIT @limit
        """
        
        # Define the query parameters
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("limit", "INT64", limit),
            ]
        )
        
        # Initialize BigQuery client
        bq_client = bigquery.Client()
        
        # Execute the query with parameters
        query_job = bq_client.query(query, job_config=job_config)
        
        # Ensure the query completes successfully
        query_job.result()
        
        # Fetch and process the results with validation
        results = []
        for row in query_job:
            row_dict = dict(row)
            
            # Validate required fields
            if not all(key in row_dict for key in ['game_date', 'home_team_name', 'away_team_name', 'status']):
                logging.warning("Skipping record with missing required fields")
                continue
                
            # Validate and convert date
            try:
                if row_dict['game_date']:
                    row_dict['game_date'] = row_dict['game_date'].isoformat()
                else:
                    logging.warning("Skipping record with null game date")
                    continue
            except (AttributeError, ValueError) as e:
                logging.warning(f"Invalid date format: {e}")
                continue
                
            # Validate team names
            if not row_dict['home_team_name'] or not row_dict['away_team_name']:
                logging.warning("Skipping record with missing team name")
                continue
                
            # Validate status
            if row_dict['status'] != 'Scheduled':
                logging.warning(f"Unexpected game status: {row_dict['status']}")
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
        logging.error(f"Unexpected error in fetch_upcoming_games: {e}")
        raise


def fetch_player_season_stats(season: int = 2024, limit: int = 1000) -> list:
    """
    Fetches player season statistics for a given MLB season.
    
    Args:
        season (int): The season year to fetch stats for. Defaults to 2024.
        limit (int): The maximum number of results to retrieve. Defaults to 1000.
    
    Returns:
        list: A list of dictionaries containing player stats, including first name, last name, team, 
              homeruns, RBI, runs, and stolen bases.
    
    Raises:
        BigQueryError: If there's an issue with the BigQuery execution.
        Exception: For other unexpected errors.
    """
    try:
        # Define the query with parameterized values
        query = """
        SELECT 
          first_name,
          last_name,
          home_run,
          b_rbi,
          r_run
        FROM 
          `gem-rush-007.mlb_data.player_stats`
        WHERE 
          year = @season
        LIMIT @limit
        """

        # Define the query parameters
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("season", "INT64", season),
                bigquery.ScalarQueryParameter("limit", "INT64", limit),
            ]
        )
        
        # Initialize BigQuery client
        bq_client = bigquery.Client()
        
        # Execute the query with parameters
        query_job = bq_client.query(query, job_config=job_config)
        
        # Ensure the query completes successfully
        query_job.result()
        
        # Fetch and process the results
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
        logging.error(f"Unexpected error in fetch_player_season_stats: {e}")
        raise


def fetch_top_batting_stats(season: int = 2024, limit: int = 1000) -> list:
    """
    Fetches player batting statistics for a given MLB season (year), ordered by batting average.
    
    Args:
        season (int): The season year to fetch stats for. Defaults to 2024.
        limit (int): The maximum number of results to retrieve. Defaults to 1000.
    
    Returns:
        list: A list of dictionaries containing player stats, including first name, last name, team, 
              batting average, on-base percentage, slugging percentage, and OPS.
    
    Raises:
        BigQueryError: If there's an issue with the BigQuery execution.
        Exception: For other unexpected errors.
    """
    try:
        # Define the query with parameterized values
        query = """
        SELECT 
          first_name,
          last_name,
          batting_avg,
          on_base_percent,
          slg_percent,
          on_base_plus_slg
        FROM 
          `gem-rush-007.mlb_data.player_stats`
        WHERE 
          year = @season
        ORDER BY 
          batting_avg DESC
        LIMIT @limit
        """
        
        # Define the query parameters
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("season", "INT64", season),
                bigquery.ScalarQueryParameter("limit", "INT64", limit),
            ]
        )
        
        # Initialize BigQuery client
        bq_client = bigquery.Client()
        
        # Execute the query with parameters
        query_job = bq_client.query(query, job_config=job_config)
        
        # Ensure the query completes successfully
        query_job.result()
        
        # Fetch and process the results
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
        logging.error(f"Unexpected error in fetch_top_batting_stats: {e}")
        raise


def fetch_top_exit_velocity(limit: int = 10) -> list:
    """
    Fetches player statistics ordered by exit velocity in descending order.

    Args:
        limit (int): The maximum number of results to retrieve. Defaults to 10.

    Returns:
        list: A list of dictionaries containing player stats, including first name, 
              last name, and exit velocity average.

    Raises:
        BigQueryError: If there's an issue with the BigQuery execution.
        Exception: For other unexpected errors.
    """
    try:
        # Define the query with parameterized values
        query = """
        SELECT  
            first_name,
            last_name,
            exit_velocity_avg
        FROM 
            `gem-rush-007.mlb_data.player_stats`
        WHERE 
            exit_velocity_avg IS NOT NULL
        ORDER BY 
            exit_velocity_avg DESC
        LIMIT @limit
        """
        
        # Define the query parameters
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("limit", "INT64", limit),
            ]
        )
        
        # Initialize BigQuery client
        bq_client = bigquery.Client()
        
        # Execute the query with parameters
        query_job = bq_client.query(query, job_config=job_config)
        
        # Ensure the query completes successfully
        query_job.result()
        
        # Fetch and process the results
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
        logging.error(f"Unexpected error in fetch_top_exit_velocity: {e}")
        raise

def fetch_top_xwoba(limit: int = 10) -> list:
    """
    Fetches player statistics ordered by xwOBA (expected weighted on-base average) in descending order.

    Args:
        limit (int): The maximum number of results to retrieve. Defaults to 10.

    Returns:
        list: A list of dictionaries containing player stats, including first name, 
              last name, and xwOBA.

    Raises:
        BigQueryError: If there's an issue with the BigQuery execution.
        Exception: For other unexpected errors.
    """
    try:
        # Define the query with parameterized values
        query = """
        SELECT 
            first_name,
            last_name,
            xwoba
        FROM 
            `gem-rush-007.mlb_data.player_stats`
        WHERE 
            xwoba IS NOT NULL
        ORDER BY 
            xwoba DESC
        LIMIT @limit
        """
        
        # Define the query parameters
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("limit", "INT64", limit),
            ]
        )
        
        # Initialize BigQuery client
        bq_client = bigquery.Client()
        
        # Execute the query with parameters
        query_job = bq_client.query(query, job_config=job_config)
        
        # Ensure the query completes successfully
        query_job.result()
        
        # Fetch and process the results
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
        logging.error(f"Unexpected error in fetch_top_xwoba: {e}")
        raise


def fetch_top_players_by_xwoba(limit: int = 10) -> list:
    """
    Fetches players ordered by xwOBA (expected weighted on-base average) in descending order.

    Args:
        limit (int): Maximum number of results to return. Must be between 1 and 100.

    Returns:
        list: A list of dictionaries containing player stats, including first name, 
              last name, and xwOBA value.

    Raises:
        ValueError: If the limit parameter is invalid.
        BigQueryError: If there's an issue with the BigQuery execution.
        Exception: For other unexpected errors.
    """
    # Input validation
    if not isinstance(limit, int):
        raise ValueError("Limit must be an integer")
    if limit < 1 or limit > 100:
        raise ValueError("Limit must be between 1 and 100")

    try:
        # Define the query with parameterized limit
        query = """
        SELECT 
            first_name,
            last_name,
            xwoba
        FROM 
            `gem-rush-007.mlb_data.player_stats`
        WHERE 
            xwoba IS NOT NULL
            AND first_name IS NOT NULL
            AND last_name IS NOT NULL
        ORDER BY 
            xwoba DESC
        LIMIT @limit
        """
        
        # Define the query parameters
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("limit", "INT64", limit),
            ]
        )
        
        # Initialize BigQuery client
        bq_client = bigquery.Client()
        
        # Execute the query with parameters
        query_job = bq_client.query(query, job_config=job_config)
        
        # Ensure the query completes successfully
        query_job.result()
        
        # Fetch and process the results with validation
        results = []
        for row in query_job:
            row_dict = dict(row)
            
            # Validate required fields
            if not row_dict.get('first_name') or not row_dict.get('last_name'):
                logging.warning("Skipping record with missing name information")
                continue
            
            # Validate xwoba value
            try:
                xwoba = float(row_dict.get('xwoba', 0))
                if not 0 <= xwoba <= 1:  # xwOBA is typically between 0 and 1
                    logging.warning(f"Invalid xwOBA value ({xwoba}) for {row_dict['first_name']} {row_dict['last_name']}")
                    continue
                row_dict['xwoba'] = xwoba
            except (TypeError, ValueError) as e:
                logging.warning(f"Invalid xwOBA value for {row_dict.get('first_name')} {row_dict.get('last_name')}: {e}")
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
        logging.error(f"Unexpected error in fetch_top_players_by_xwoba: {e}")
        raise


def fetch_top_players_by_hard_hit(limit: int = 10) -> list:
    """
    Fetches players ordered by hard hit percentage in descending order.

    Args:
        limit (int): Maximum number of results to return. Must be between 1 and 100.

    Returns:
        list: A list of dictionaries containing player stats, including first name, 
              last name, and hard hit percentage.

    Raises:
        ValueError: If the limit parameter is invalid.
        BigQueryError: If there's an issue with the BigQuery execution.
        Exception: For other unexpected errors.
    """
    # Input validation
    if not isinstance(limit, int):
        raise ValueError("Limit must be an integer")
    if limit < 1 or limit > 100:
        raise ValueError("Limit must be between 1 and 100")

    try:
        # Define the query with parameterized limit
        query = """
        SELECT 
            first_name,
            last_name,
            hard_hit_percent
        FROM 
            `gem-rush-007.mlb_data.player_stats`
        WHERE 
            hard_hit_percent IS NOT NULL
            AND first_name IS NOT NULL
            AND last_name IS NOT NULL
            AND hard_hit_percent > 0  -- Ensure valid percentage
            AND hard_hit_percent <= 100  -- Ensure valid percentage
        ORDER BY 
            hard_hit_percent DESC
        LIMIT @limit
        """
        
        # Define the query parameters
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("limit", "INT64", limit),
            ]
        )
        
        # Initialize BigQuery client
        bq_client = bigquery.Client()
        
        # Execute the query with parameters
        query_job = bq_client.query(query, job_config=job_config)
        
        # Ensure the query completes successfully
        query_job.result()
        
        # Fetch and process the results with validation
        results = []
        for row in query_job:
            row_dict = dict(row)
            
            # Validate required fields
            if not row_dict.get('first_name') or not row_dict.get('last_name'):
                logging.warning("Skipping record with missing name information")
                continue
            
            # Validate hard hit percentage
            try:
                hard_hit_pct = float(row_dict.get('hard_hit_percent', 0))
                if not 0 <= hard_hit_pct <= 100:  # Valid percentage range
                    logging.warning(
                        f"Invalid hard hit percentage ({hard_hit_pct}) for "
                        f"{row_dict['first_name']} {row_dict['last_name']}"
                    )
                    continue
                row_dict['hard_hit_percent'] = round(hard_hit_pct, 2)  # Round to 2 decimal places
            except (TypeError, ValueError) as e:
                logging.warning(
                    f"Invalid hard hit percentage value for "
                    f"{row_dict.get('first_name')} {row_dict.get('last_name')}: {e}"
                )
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
        logging.error(f"Unexpected error in fetch_top_players_by_hard_hit: {e}")
        raise


def fetch_top_players_by_whiff_percent(limit: int = 10) -> list:
    """
    Fetches players ordered by whiff percentage in ascending order (lower is better).

    Args:
        limit (int): Maximum number of results to return. Must be between 1 and 100.

    Returns:
        list: A list of dictionaries containing player stats, including first name, 
              last name, and whiff percentage (percentage of swings that miss).

    Raises:
        ValueError: If the limit parameter is invalid.
        BigQueryError: If there's an issue with the BigQuery execution.
        Exception: For other unexpected errors.
    """
    # Input validation
    if not isinstance(limit, int):
        raise ValueError("Limit must be an integer")
    if limit < 1 or limit > 100:
        raise ValueError("Limit must be between 1 and 100")

    try:
        # Define the query with parameterized limit
        query = """
        SELECT 
            first_name,
            last_name,
            whiff_percent
        FROM 
            `gem-rush-007.mlb_data.player_stats`
        WHERE 
            whiff_percent IS NOT NULL
            AND first_name IS NOT NULL
            AND last_name IS NOT NULL
            AND whiff_percent >= 0  -- Ensure valid percentage
            AND whiff_percent <= 100  -- Ensure valid percentage
        ORDER BY 
            whiff_percent ASC  -- Lower whiff percentage is better
        LIMIT @limit
        """
        
        # Define the query parameters
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("limit", "INT64", limit),
            ]
        )
        
        # Initialize BigQuery client
        bq_client = bigquery.Client()
        
        # Execute the query with parameters
        query_job = bq_client.query(query, job_config=job_config)
        
        # Ensure the query completes successfully
        query_job.result()
        
        # Fetch and process the results with validation
        results = []
        for row in query_job:
            row_dict = dict(row)
            
            # Validate required fields
            if not row_dict.get('first_name') or not row_dict.get('last_name'):
                logging.warning("Skipping record with missing name information")
                continue
            
            # Validate whiff percentage
            try:
                whiff_pct = float(row_dict.get('whiff_percent', 0))
                if not 0 <= whiff_pct <= 100:  # Valid percentage range
                    logging.warning(
                        f"Invalid whiff percentage ({whiff_pct}) for "
                        f"{row_dict['first_name']} {row_dict['last_name']}"
                    )
                    continue
                row_dict['whiff_percent'] = round(whiff_pct, 2)  # Round to 2 decimal places
            except (TypeError, ValueError) as e:
                logging.warning(
                    f"Invalid whiff percentage value for "
                    f"{row_dict.get('first_name')} {row_dict.get('last_name')}: {e}"
                )
                continue
            
            # Add a calculated field for display purposes
            row_dict['contact_percent'] = round(100 - whiff_pct, 2)  # Calculate contact percentage
            
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
        logging.error(f"Unexpected error in fetch_top_players_by_whiff_percent: {e}")
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


def fetch_top_players_by_barrel_rate(limit: int = 10) -> list:
    """
    Fetches players ordered by barrel batted rate in descending order (higher is better).

    Args:
        limit (int): Maximum number of results to return. Must be between 1 and 100.

    Returns:
        list: A list of dictionaries containing player stats, including first name, 
              last name, and barrel batted rate (percentage of batted balls that are barreled).

    Raises:
        ValueError: If the limit parameter is invalid.
        BigQueryError: If there's an issue with the BigQuery execution.
        Exception: For other unexpected errors.
    """
    # Input validation
    if not isinstance(limit, int):
        raise ValueError("Limit must be an integer")
    if limit < 1 or limit > 100:
        raise ValueError("Limit must be between 1 and 100")

    try:
        # Define the query with parameterized limit
        query = """
        SELECT 
            first_name,
            last_name,
            barrel_batted_rate,
    
        FROM 
            `gem-rush-007.mlb_data.player_stats`
        WHERE 
            barrel_batted_rate IS NOT NULL
            AND first_name IS NOT NULL
            AND last_name IS NOT NULL
        ORDER BY 
            barrel_batted_rate DESC
        LIMIT @limit
        """
        
        # Define the query parameters
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("limit", "INT64", limit),
            ]
        )
        
        # Initialize BigQuery client
        bq_client = bigquery.Client()
        
        # Execute the query with parameters
        query_job = bq_client.query(query, job_config=job_config)
        
        # Ensure the query completes successfully
        query_job.result()
        
        # Fetch and process the results with validation
        results = []
        for row in query_job:
            row_dict = dict(row)
            
            # Validate required fields
            if not row_dict.get('first_name') or not row_dict.get('last_name'):
                logging.warning("Skipping record with missing name information")
                continue
            
            # Validate barrel rate
            try:
                barrel_rate = float(row_dict.get('barrel_batted_rate', 0))
                if not 0 <= barrel_rate <= 100:  # Valid percentage range
                    logging.warning(
                        f"Invalid barrel rate ({barrel_rate}) for "
                        f"{row_dict['first_name']} {row_dict['last_name']}"
                    )
                    continue
                row_dict['barrel_batted_rate'] = round(barrel_rate, 2)  # Round to 2 decimal places
                       
            except (TypeError, ValueError) as e:
                logging.warning(
                    f"Invalid data for {row_dict.get('first_name')} "
                    f"{row_dict.get('last_name')}: {e}"
                )
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
        logging.error(f"Unexpected error in fetch_top_players_by_barrel_rate: {e}")
        raise



def fetch_top_players_by_sweet_spot(limit: int = 10) -> list:
    """
    Fetches players ordered by sweet spot percentage in descending order (higher is better).

    Args:
        limit (int): Maximum number of results to return. Must be between 1 and 100.

    Returns:
        list: A list of dictionaries containing player stats, including first name, 
              last name, and sweet spot percentage.

    Raises:
        ValueError: If the limit parameter is invalid.
        BigQueryError: If there's an issue with the BigQuery execution.
        Exception: For other unexpected errors.
    """
    # Input validation
    if not isinstance(limit, int):
        raise ValueError("Limit must be an integer")
    if limit < 1 or limit > 100:
        raise ValueError("Limit must be between 1 and 100")

    try:
        # Define the query with parameterized limit
        query = """
        SELECT 
            first_name,
            last_name,
            sweet_spot_percent
        FROM 
            `gem-rush-007.mlb_data.player_stats`
        WHERE 
            sweet_spot_percent IS NOT NULL
        ORDER BY 
            sweet_spot_percent DESC
        LIMIT @limit
        """
        
        # Define the query parameters
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("limit", "INT64", limit),
            ]
        )
        
        # Initialize BigQuery client
        bq_client = bigquery.Client()
        
        # Execute the query with parameters
        query_job = bq_client.query(query, job_config=job_config)
        
        # Fetch and process the results with validation
        results = []
        for row in query_job:
            row_dict = dict(row)
            
            # Validate required fields
            if not row_dict.get('first_name') or not row_dict.get('last_name'):
                logging.warning("Skipping record with missing name information")
                continue
            
            # Validate sweet spot percentage
            try:
                sweet_spot = float(row_dict.get('sweet_spot_percent', 0))
                if not 0 <= sweet_spot <= 100:  # Valid percentage range
                    logging.warning(
                        f"Invalid sweet spot percentage ({sweet_spot}) for "
                        f"{row_dict['first_name']} {row_dict['last_name']}"
                    )
                    continue
                row_dict['sweet_spot_percent'] = round(sweet_spot, 2)  # Round to 2 decimal places
                       
            except (TypeError, ValueError) as e:
                logging.warning(
                    f"Invalid data for {row_dict.get('first_name')} "
                    f"{row_dict.get('last_name')}: {e}"
                )
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
        logging.error(f"Unexpected error in fetch_top_players_by_sweet_spot: {e}")
        raise


def fetch_top_players_by_xslg(limit: int = 10) -> list:
    """
    Fetches players ordered by expected slugging percentage (xSLG) in descending order.

    Args:
        limit (int): Maximum number of results to return. Must be between 1 and 100.

    Returns:
        list: A list of dictionaries containing player stats, including first name, 
              last name, and expected slugging percentage (xSLG).

    Raises:
        ValueError: If the limit parameter is invalid.
        BigQueryError: If there's an issue with the BigQuery execution.
        Exception: For other unexpected errors.
    """
    # Input validation
    if not isinstance(limit, int):
        raise ValueError("Limit must be an integer")
    if limit < 1 or limit > 100:
        raise ValueError("Limit must be between 1 and 100")

    try:
        # Define the query with parameterized limit
        query = """
        SELECT 
            first_name,
            last_name,
            xslg
        FROM 
            `gem-rush-007.mlb_data.player_stats`
        WHERE 
            xslg IS NOT NULL
        ORDER BY 
            xslg DESC
        LIMIT @limit
        """
        
        # Define the query parameters
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("limit", "INT64", limit),
            ]
        )
        
        # Initialize BigQuery client
        bq_client = bigquery.Client()
        
        # Execute the query with parameters
        query_job = bq_client.query(query, job_config=job_config)
        
        # Fetch and process the results with validation
        results = []
        for row in query_job:
            row_dict = dict(row)
            
            # Validate required fields
            if not row_dict.get('first_name') or not row_dict.get('last_name'):
                logging.warning("Skipping record with missing name information")
                continue
            
            # Validate xSLG
            try:
                xslg = float(row_dict.get('xslg', 0))
                if not 0 <= xslg <= 5:  # Reasonable range for slugging percentage
                    logging.warning(
                        f"Invalid xSLG value ({xslg}) for "
                        f"{row_dict['first_name']} {row_dict['last_name']}"
                    )
                    continue
                row_dict['xslg'] = round(xslg, 3)  # Round to 3 decimal places
                       
            except (TypeError, ValueError) as e:
                logging.warning(
                    f"Invalid data for {row_dict.get('first_name')} "
                    f"{row_dict.get('last_name')}: {e}"
                )
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
        logging.error(f"Unexpected error in fetch_top_players_by_xslg: {e}")
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


def fetch_top_players_by_xba(limit: int = 10) -> list:
    """
    Fetches players ordered by expected batting average (xBA) in descending order.

    Args:
        limit (int): Maximum number of results to return. Must be between 1 and 100.

    Returns:
        list: A list of dictionaries containing player stats, including first name, 
              last name, and expected batting average (xBA).

    Raises:
        ValueError: If the limit parameter is invalid.
        BigQueryError: If there's an issue with the BigQuery execution.
        Exception: For other unexpected errors.
    """
    # Input validation
    if not isinstance(limit, int):
        raise ValueError("Limit must be an integer")
    if limit < 1 or limit > 100:
        raise ValueError("Limit must be between 1 and 100")

    try:
        # Define the query with parameterized limit
        query = """
        SELECT 
            first_name,
            last_name,
            xba
        FROM 
            `gem-rush-007.mlb_data.player_stats`
        WHERE 
            xba IS NOT NULL
        ORDER BY 
            xba DESC
        LIMIT @limit
        """
        
        # Define the query parameters
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("limit", "INT64", limit),
            ]
        )
        
        # Initialize BigQuery client
        bq_client = bigquery.Client()
        
        # Execute the query with parameters
        query_job = bq_client.query(query, job_config=job_config)
        
        # Fetch and process the results with validation
        results = []
        for row in query_job:
            row_dict = dict(row)
            
            # Validate required fields
            if not row_dict.get('first_name') or not row_dict.get('last_name'):
                logging.warning("Skipping record with missing name information")
                continue
            
            # Validate xBA
            try:
                xba = float(row_dict.get('xba', 0))
                if not 0 <= xba <= 1:  # Valid batting average range
                    logging.warning(
                        f"Invalid xBA value ({xba}) for "
                        f"{row_dict['first_name']} {row_dict['last_name']}"
                    )
                    continue
                row_dict['xba'] = round(xba, 3)  # Round to 3 decimal places
                       
            except (TypeError, ValueError) as e:
                logging.warning(
                    f"Invalid data for {row_dict.get('first_name')} "
                    f"{row_dict.get('last_name')}: {e}"
                )
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
        logging.error(f"Unexpected error in fetch_top_players_by_xba: {e}")
        raise


def fetch_top_players_by_edge_percent(limit: int = 10) -> list:
    """
    Fetches players ordered by edge percentage in descending order.

    Args:
        limit (int): Maximum number of results to return. Must be between 1 and 100.

    Returns:
        list: A list of dictionaries containing player stats, including first name, 
              last name, and edge percentage.

    Raises:
        ValueError: If the limit parameter is invalid.
        BigQueryError: If there's an issue with the BigQuery execution.
        Exception: For other unexpected errors.
    """
    # Input validation
    if not isinstance(limit, int):
        raise ValueError("Limit must be an integer")
    if limit < 1 or limit > 100:
        raise ValueError("Limit must be between 1 and 100")

    try:
        # Define the query with parameterized limit
        query = """
        SELECT 
            first_name,
            last_name,
            edge_percent
        FROM 
            `gem-rush-007.mlb_data.player_stats`
        WHERE 
            edge_percent IS NOT NULL
        ORDER BY 
            edge_percent DESC
        LIMIT @limit
        """
        
        # Define the query parameters
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("limit", "INT64", limit),
            ]
        )
        
        # Initialize BigQuery client
        bq_client = bigquery.Client()
        
        # Execute the query with parameters
        query_job = bq_client.query(query, job_config=job_config)
        
        # Fetch and process the results with validation
        results = []
        for row in query_job:
            row_dict = dict(row)
            
            # Validate required fields
            if not row_dict.get('first_name') or not row_dict.get('last_name'):
                logging.warning("Skipping record with missing name information")
                continue
            
            # Validate edge percentage
            try:
                edge_pct = float(row_dict.get('edge_percent', 0))
                if not 0 <= edge_pct <= 100:  # Valid percentage range
                    logging.warning(
                        f"Invalid edge percentage ({edge_pct}) for "
                        f"{row_dict['first_name']} {row_dict['last_name']}"
                    )
                    continue
                row_dict['edge_percent'] = round(edge_pct, 2)  # Round to 2 decimal places
                       
            except (TypeError, ValueError) as e:
                logging.warning(
                    f"Invalid data for {row_dict.get('first_name')} "
                    f"{row_dict.get('last_name')}: {e}"
                )
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
        logging.error(f"Unexpected error in fetch_top_players_by_edge_percent: {e}")
        raise


def fetch_top_players_by_walk_rate(limit: int = 10) -> list:
    """
    Fetches players ordered by walk rate (bb_percent) in descending order.

    Args:
        limit (int): Maximum number of results to return. Must be between 1 and 100.

    Returns:
        list: A list of dictionaries containing player stats, including first name, 
              last name, walk rate, and strikeout rate.

    Raises:
        ValueError: If the limit parameter is invalid.
        BigQueryError: If there's an issue with the BigQuery execution.
        Exception: For other unexpected errors.
    """
    # Input validation
    if not isinstance(limit, int):
        raise ValueError("Limit must be an integer")
    if limit < 1 or limit > 100:
        raise ValueError("Limit must be between 1 and 100")

    try:
        # Define the query with parameterized limit
        query = """
        SELECT 
            first_name,
            last_name,
            bb_percent AS Walk_Rate,
            k_percent AS Strikeout_Rate
        FROM 
            `gem-rush-007.mlb_data.player_stats`
        WHERE 
            bb_percent IS NOT NULL AND k_percent IS NOT NULL
        ORDER BY 
            CAST(bb_percent AS FLOAT64) DESC
        LIMIT @limit
        """
        
        # Define the query parameters
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("limit", "INT64", limit),
            ]
        )
        
        # Initialize BigQuery client
        bq_client = bigquery.Client()
        
        # Execute the query with parameters
        query_job = bq_client.query(query, job_config=job_config)
        
        # Fetch and process the results with validation
        results = []
        for row in query_job:
            row_dict = dict(row)
            
            # Validate required fields
            if not row_dict.get('first_name') or not row_dict.get('last_name'):
                logging.warning("Skipping record with missing name information")
                continue
            
            # Validate percentage fields
            try:
                walk_rate = float(row_dict.get('Walk_Rate', 0))
                strikeout_rate = float(row_dict.get('Strikeout_Rate', 0))
                
                if not (0 <= walk_rate <= 100 and 0 <= strikeout_rate <= 100):
                    logging.warning(
                        f"Invalid percentage values for {row_dict['first_name']} "
                        f"{row_dict['last_name']}: Walk Rate={walk_rate}, "
                        f"Strikeout Rate={strikeout_rate}"
                    )
                    continue
                
                row_dict['Walk_Rate'] = round(walk_rate, 2)
                row_dict['Strikeout_Rate'] = round(strikeout_rate, 2)
                       
            except (TypeError, ValueError) as e:
                logging.warning(
                    f"Invalid data for {row_dict.get('first_name')} "
                    f"{row_dict.get('last_name')}: {e}"
                )
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
        logging.error(f"Unexpected error in fetch_top_players_by_walk_rate: {e}")
        raise

def fetch_top_players_by_launch_angle(limit: int = 10) -> list:
    """
    Fetches players ordered by launch angle average in descending order.

    Args:
        limit (int): Maximum number of results to return. Must be between 1 and 100.

    Returns:
        list: A list of dictionaries containing player stats, including first name, 
              last name, and launch angle average.

    Raises:
        ValueError: If the limit parameter is invalid.
        BigQueryError: If there's an issue with the BigQuery execution.
        Exception: For other unexpected errors.
    """
    # Input validation
    if not isinstance(limit, int):
        raise ValueError("Limit must be an integer")
    if limit < 1 or limit > 100:
        raise ValueError("Limit must be between 1 and 100")

    try:
        # Define the query with parameterized limit
        query = """
        SELECT 
            first_name,
            last_name,
            launch_angle_avg
        FROM 
            `gem-rush-007.mlb_data.player_stats`
        WHERE 
            launch_angle_avg IS NOT NULL
        ORDER BY 
            launch_angle_avg DESC
        LIMIT @limit
        """
        
        # Define the query parameters
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("limit", "INT64", limit),
            ]
        )
        
        # Initialize BigQuery client
        bq_client = bigquery.Client()
        
        # Execute the query with parameters
        query_job = bq_client.query(query, job_config=job_config)
        
        # Fetch and process the results with validation
        results = []
        for row in query_job:
            row_dict = dict(row)
            
            # Validate required fields
            if not row_dict.get('first_name') or not row_dict.get('last_name'):
                logging.warning("Skipping record with missing name information")
                continue
            
            # Validate launch angle
            try:
                launch_angle = float(row_dict.get('launch_angle_avg', 0))
                # Typical launch angles in baseball are between -90 and 90 degrees
                if not -90 <= launch_angle <= 90:
                    logging.warning(
                        f"Invalid launch angle ({launch_angle}) for "
                        f"{row_dict['first_name']} {row_dict['last_name']}"
                    )
                    continue
                row_dict['launch_angle_avg'] = round(launch_angle, 2)  # Round to 2 decimal places
                       
            except (TypeError, ValueError) as e:
                logging.warning(
                    f"Invalid data for {row_dict.get('first_name')} "
                    f"{row_dict.get('last_name')}: {e}"
                )
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
        logging.error(f"Unexpected error in fetch_top_players_by_launch_angle: {e}")
        raise

def fetch_roster_players(team_name: str, limit: int = 1000) -> list:
    """
    Fetches active players from any MLB team's roster.

    Args:
        team_name (str): Team name from TEAMS dictionary
        limit (int): Maximum number of results to return. Must be between 1 and 1000.

    Returns:
        list: A list of dictionaries containing player names.

    Raises:
        ValueError: If the limit parameter is invalid or team_name not found.
        BigQueryError: If there's an issue with the BigQuery execution.
        Exception: For other unexpected errors.
    """
    # Input validation
    if not isinstance(limit, int):
        raise ValueError("Limit must be an integer")
    if limit < 1 or limit > 1000:
        raise ValueError("Limit must be between 1 and 1000")

    # Get team-specific table name
    table_name = _get_table_name(team_name)
    
    try:
        print(table_name)
        # Define the query with parameterized limit
        query = f"""
        SELECT 
            player_id,
            full_name,
            status,
            position,
            jersey_number
        FROM 
            {table_name}.roster
        WHERE
            status = "Active"
        ORDER BY
            full_name ASC
        LIMIT @limit
        """
        
        # Define the query parameters
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("limit", "INT64", limit),
            ]
        )
        
        # Initialize BigQuery client
        bq_client = bigquery.Client()
        
        # Execute the query with parameters
        query_job = bq_client.query(query, job_config=job_config)
        
        # Fetch and process the results with validation
        results = []
        for row in query_job:
            row_dict = dict(row)
            
            # Validate required fields
            if not row_dict.get('full_name'):
                logging.warning("Skipping record with missing name information")
                continue
            
            # Clean and validate additional fields
            if row_dict.get('jersey_number'):
                try:
                    row_dict['jersey_number'] = int(row_dict['jersey_number'])
                except (TypeError, ValueError):
                    row_dict['jersey_number'] = None
                    logging.warning(f"Invalid jersey number for player {row_dict['full_name']}")
            
            results.append(row_dict)
        
        if not results:
            logging.warning(f"Query returned no active roster players for {team_name}")
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
        logging.error(f"Unexpected error in fetch_roster_players_generic for {team_name}: {e}")
        raise


def fetch_players_by_names(player_names: list) -> list:
    """
    Fetches player images based on provided list of names.
    
    Args:
        player_names: List of player names
        
    Returns:
        list: List of dictionaries containing player info and image URLs
    """
    try:
        query = """
        SELECT 
            player_name,
            team,
            signed_url
        FROM 
            `gem-rush-007.mlb_data.player_names`
        WHERE 
            LOWER(player_name) IN UNNEST(@player_names)
        """
        
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ArrayQueryParameter(
                    "player_names", 
                    "STRING", 
                    [name.lower() for name in player_names]
                ),
            ]
        )
        
        bq_client = bigquery.Client()
        query_job = bq_client.query(query, job_config=job_config)
        
        results = []
        for row in query_job:
            results.append({
                'player_name': row.player_name,
                'team': row.team,
                'image_url': row.signed_url
            })
            
        if not results:
            logging.warning(f"No players found with names: {player_names}")
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
        logging.error(f"Error fetching player images: {e}")
        raise


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

def fetch_all_mlb_teams(limit: int = 1000) -> list:
    """
    Fetches all MLB teams from the teams table.

    Args:
        limit (int, optional): Maximum number of teams to return. Defaults to 1000.

    Returns:
        list: A list of dictionaries containing team information
    """
    try:
        query = """
        SELECT
        
            name,
            
        FROM
            `gem-rush-007.teams.teams_data`
        WHERE
            active = true
            AND season = 2024
        ORDER BY 
            league_name,
            division_name,
            team_name
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
        logging.error(f"Error in fetch_all_mlb_teams: {e}")
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
        LIMIT 5;
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
        data = [dict(row) for row in results]
       
        return  data
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

def generate_mlb_analysis(contents: str) -> dict:
   
    client = genai.Client(vertexai=True, project="gem-rush-007", location="us-central1")
    MODEL_ID = "gemini-2.0-flash-exp"

    # Structure the prompt to explicitly request tool usage
    structured_prompt = f"""
    To answer this question, please:
    1. Use the appropriate tools to fetch and analyze the data
    2. Analyze the data and provide insights
   
    Question: {contents}

    """

    try:
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=structured_prompt,
            config=GenerateContentConfig(
                tools=[
                    # Prioritize the player_plays function
                    fetch_player_plays,
                    # ... other tools ...
                    get_player_highest_ops,
                    analyze_position_slugging,
                    analyze_team_strikeouts,
                    analyze_monthly_home_runs,
                    analyze_homerun_win_correlation,
                    analyze_position_ops_percentile,
                    analyze_position_batting_efficiency,
                    analyze_team_weight_trends,
                    analyze_stolen_base_efficiency,
                    analyze_ops_percentile_trends,
                    analyze_home_scoring_vs_high_away_scores,
                    analyze_near_cycle_players,
                    get_player_recent_streaks,
                    compare_rookie_season_to_veteran,
                    identify_undervalued_players,
                    predict_matchup_outcome_by_stats,
                    fetch_team_performance_by_venue,
                    fetch_top_batting_stats,
                    fetch_top_exit_velocity,
                    fetch_top_players_by_xwoba,
                    fetch_top_players_by_hard_hit,
                    fetch_top_players_by_whiff_percent,
                    fetch_top_players_by_barrel_rate,
                    fetch_top_players_by_sweet_spot,
                    fetch_top_players_by_xslg,
                    fetch_top_players_by_xba,
                    fetch_top_players_by_edge_percent,
                    fetch_top_players_by_walk_rate,
                    fetch_top_players_by_launch_angle,
                    fetch_roster_players,
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
                    fetch_all_mlb_teams,
                    _get_table_name,
                ],
                temperature=0,
            ),
        )

        text_response = response.text
        return text_response

    except Exception as e:
        logging.error(f"Error in generate_mlb_analysis: {e}")
        return {
            "text": str(e),
            "iframe_url": None
        }