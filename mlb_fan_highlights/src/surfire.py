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

PROJECT_ID = "gem-creation"  # Replace with your actual Google Cloud project ID
os.environ["GOOGLE_CLOUD_PROJECT"] = PROJECT_ID # set this environment variable to your project ID
bq_client = bigquery.Client(project=PROJECT_ID)

client = genai.Client(vertexai=True, project="gem-creation", location="us-central1")
MODEL_ID = "gemini-2.0-flash-exp"  # @param {type: "string"}

def get_player_highest_ops(season: int) -> str:
    """Returns the player with the highest OPS (On-base Plus Slugging) for a given season.

    Args:
        season: The MLB season year (e.g., 2020)
    Returns:
        str: Description of the player and their OPS
    """
    query = """
    SELECT
        first_name,
        last_name,
        on_base_plus_slugging,
        team_name
    FROM
        `mlb_data.combined_player_stats`
    WHERE
        season = @season
    QUALIFY
        ROW_NUMBER() OVER (PARTITION BY season ORDER BY on_base_plus_slugging DESC) = 1
    """
    # Execute query and return formatted result
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("season", "INTEGER", season)
        ]
    )
    query_job = bq_client.query(query, job_config=job_config)
    results = query_job.result()
    return [dict(row) for row in results][0]  # Return first row as dict

def calculate_team_home_advantage(team_name: str) -> str:
    """Calculates the home field advantage statistics for a specific team.
    
    Args:
        team_name: The name of the MLB team
    Returns:
        str: Analysis of the team's home field advantage
    """
    query = """
    SELECT
        team_name,
        COUNTIF(home_score > away_score) * 100.0 / COUNT(*) AS home_win_percentage,
        AVG(home_score) AS avg_home_runs,
        AVG(away_score) AS avg_away_runs
    FROM
        `mlb_data.combined_player_stats`
    WHERE
        team_name = @team_name
    GROUP BY
        team_name
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("team_name", "STRING", team_name)
        ]
    )
    query_job = bq_client.query(query, job_config=job_config)
    results = query_job.result()
    return [dict(row) for row in results][0]  # Return first row as dict

def analyze_player_performance(player_first_name: str, player_last_name: str) -> str:
    """Analyzes a player's career statistics and performance trends.
    
    Args:
        player_first_name: Player's first name
        player_last_name: Player's last name
    Returns:
        str: Comprehensive analysis of the player's performance
    """
    query = """
    SELECT
        AVG(batting_average) as career_avg,
        SUM(homeruns) as total_hr,
        SUM(stolen_bases) as total_sb,
        COUNT(DISTINCT season) as seasons_played
    FROM
        `mlb_data.combined_player_stats`
    WHERE
        first_name = @player_first_name
        AND last_name = @player_last_name
    """
    # Execute query and return formatted analysis
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("player_first_name", "STRING", player_first_name),
            bigquery.ScalarQueryParameter("player_last_name", "STRING", player_last_name)
        ]
    )
    query_job = bq_client.query(query, job_config=job_config)
    results = query_job.result()
    return [dict(row) for row in results][0]  # Return first row as dict

def analyze_weight_performance(min_weight: int, max_weight: int) -> dict:
    """Analyzes batting performance statistics for players within a weight range.
    
    Args:
        min_weight: Minimum player weight
        max_weight: Maximum player weight
    Returns:
        dict: Batting performance statistics
    """
    query = """
    SELECT
        AVG(batting_average) AS avg_batting_average,
        AVG(homeruns) AS avg_homeruns,
        COUNT(DISTINCT player_id) AS player_count
    FROM
        `mlb_data.combined_player_stats`
    WHERE
        weight BETWEEN @min_weight AND @max_weight
        AND at_bats >= 200
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("min_weight", "INT64", min_weight),
            bigquery.ScalarQueryParameter("max_weight", "INT64", max_weight)
        ]
    )
    query_job = bq_client.query(query, job_config=job_config)
    results = query_job.result()
    return [dict(row) for row in results][0]


def analyze_team_strikeouts(team_name: str, season: int) -> dict:
    """Analyzes team's strikeout statistics for a specific season.
    
    Args:
        team_name: The name of the MLB team
        season: The season year to analyze
    Returns:
        dict: Team strikeout statistics
    """
    query = """
    SELECT
        team_name,
        AVG(strikeouts) AS average_strikeouts_per_game,
        MAX(strikeouts) AS max_strikeouts,
        MIN(strikeouts) AS min_strikeouts
    FROM
        `mlb_data.combined_player_stats`
    WHERE
        team_name = @team_name
        AND season = @season
    GROUP BY
        team_name
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("team_name", "STRING", team_name),
            bigquery.ScalarQueryParameter("season", "INT64", season)
        ]
    )
    query_job = bq_client.query(query, job_config=job_config)
    results = query_job.result()
    return [dict(row) for row in results][0]

def analyze_monthly_home_runs(team_name: str, year: int) -> list:
    """Analyzes monthly home run trends for a specific team and year.
    
    Args:
        team_name: The name of the MLB team
        year: The year to analyze
    Returns:
        list: Monthly home run statistics
    """
    query = """
    SELECT
        team_name,
        EXTRACT(MONTH FROM game_date) AS month,
        AVG(homeruns) AS average_home_runs,
        COUNT(DISTINCT game_pk) as games_played
    FROM
        `mlb_data.combined_player_stats`
    WHERE
        team_name = @team_name
        AND EXTRACT(YEAR FROM game_date) = @year
    GROUP BY
        team_name,
        month
    ORDER BY
        month
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("team_name", "STRING", team_name),
            bigquery.ScalarQueryParameter("year", "INT64", year)
        ]
    )
    query_job = bq_client.query(query, job_config=job_config)
    results = query_job.result()
    return [dict(row) for row in results]

def analyze_position_slugging(position: str, min_games: int = 100) -> dict:
    """Analyzes slugging percentage statistics for players in a specific position.
    
    Args:
        position: The player position to analyze
        min_games: Minimum games played threshold (default 100)
    Returns:
        dict: Slugging percentage analysis for the position
    """
    query = """
    SELECT
        position_name,
        AVG(slugging_percentage) AS average_slugging_percentage,
        COUNT(DISTINCT player_id) as player_count
    FROM
        `mlb_data.combined_player_stats`
    WHERE
        position_name = @position
        AND games_played > @min_games
    GROUP BY
        position_name
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("position", "STRING", position),
            bigquery.ScalarQueryParameter("min_games", "INT64", min_games)
        ]
    )
    query_job = bq_client.query(query, job_config=job_config)
    results = query_job.result()
    return [dict(row) for row in results][0]


def analyze_position_ops_percentile(min_games: int = 50) -> list:
    """Calculates the 90th percentile OPS for each position.
    
    Args:
        min_games: Minimum games played threshold (default 50)
    Returns:
        list: 90th percentile OPS by position
    """
    query = """
    SELECT
        position_name,
        APPROX_QUANTILES(on_base_plus_slugging, 100)[OFFSET(90)] AS ops_90th_percentile,
        COUNT(DISTINCT player_id) as qualified_players
    FROM
        `mlb_data.combined_player_stats`
    WHERE
        games_played >= @min_games
    GROUP BY
        position_name
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("min_games", "INT64", min_games)
        ]
    )
    query_job = bq_client.query(query, job_config=job_config)
    return [dict(row) for row in query_job.result()]

def analyze_homerun_win_correlation(start_year: int, end_year: int) -> list:
    """Analyzes correlation between team home runs and wins by season.
    
    Args:
        start_year: Starting season year
        end_year: Ending season year
    Returns:
        list: Season-by-season correlation between home runs and wins
    """
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
        GROUP BY 1, 2
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
        GROUP BY 1, 2
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
    query_job = bq_client.query(query, job_config=job_config)
    return [dict(row) for row in query_job.result()]


def analyze_position_batting_efficiency(min_at_bats: int = 200) -> list:
    """Calculates batting efficiency metrics by position.
    
    Args:
        min_at_bats: Minimum at-bats threshold (default 200)
    Returns:
        list: Batting efficiency statistics by position
    """
    query = """
    SELECT
        position_name,
        SUM(at_bats) as total_at_bats,
        SUM(hits) as total_hits,
        SUM(at_bats) / SUM(hits) AS avg_at_bats_per_hit,
        COUNT(DISTINCT player_id) as qualified_players
    FROM
        `mlb_data.combined_player_stats`
    WHERE
        at_bats >= @min_at_bats
    GROUP BY
        position_name
    ORDER BY
        avg_at_bats_per_hit ASC
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("min_at_bats", "INT64", min_at_bats)
        ]
    )
    query_job = bq_client.query(query, job_config=job_config)
    return [dict(row) for row in query_job.result()]


def analyze_team_weight_trends(season: int) -> list:
    """Identifies teams with highest average player weights and related stats.
    
    Args:
        season: The season year to analyze
    Returns:
        list: Teams ranked by average player weight with additional metrics
    """
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
    GROUP BY
        season,
        team_name
    ORDER BY
        average_player_weight DESC
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("season", "INT64", season)
        ]
    )
    query_job = bq_client.query(query, job_config=job_config)
    return [dict(row) for row in query_job.result()]

def analyze_stolen_base_efficiency(season: int) -> list:
    """Analyzes team's stolen base efficiency in winning home games.
    
    Args:
        season: The season year to analyze
    Returns:
        list: Teams ranked by stolen base efficiency metrics
    """
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
    query_job = bq_client.query(query, job_config=job_config)
    return [dict(row) for row in query_job.result()]

def analyze_team_obp_leaders(team_name: str, min_at_bats: int = 300) -> list:
    """Identifies players with highest on-base percentage for a specific team.
    
    Args:
        team_name: The name of the MLB team
        min_at_bats: Minimum at-bats threshold (default 300)
    Returns:
        list: Players ranked by on-base percentage
    """
    query = """
    SELECT
        team_name,
        first_name,
        last_name,
        on_base_percentage,
        at_bats,
        walks,
        hits
    FROM
        `mlb_data.combined_player_stats`
    WHERE
        team_name = @team_name
        AND at_bats >= @min_at_bats
    ORDER BY
        on_base_percentage DESC
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("team_name", "STRING", team_name),
            bigquery.ScalarQueryParameter("min_at_bats", "INT64", min_at_bats)
        ]
    )
    query_job = bq_client.query(query, job_config=job_config)
    return [dict(row) for row in query_job.result()]


def analyze_ops_percentile_trends() -> list:
    """Calculates 90th percentile OPS trends across seasons.
    
    Returns:
        list: Season-by-season 90th percentile OPS values
    """
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
    return [dict(row) for row in query_job.result()]


response = client.models.generate_content(
    model=MODEL_ID,
    contents="Show the trend of 90th percentile OPS over the years.",
    config=GenerateContentConfig(
        tools=[get_player_highest_ops, 
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
               analyze_team_obp_leaders,
               analyze_ops_percentile_trends
               ],
        temperature=0,
    ),
)

print(response.text)