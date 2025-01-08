datasource - kaggle and statcast powered by google cloud

"""
nova.py: MLB Data Ingestion and BigQuery Pipeline

This module defines a pipeline for fetching real-time and historical MLB data from the MLB Stats API and loading it into a BigQuery dataset.  It handles rate limiting, data transformation, and error handling to ensure data integrity and efficient processing.

The pipeline performs the following actions:

1. **Table Creation:** Creates the BigQuery dataset (`mlb_data`) and tables if they don't exist.  The tables are structured to store team information, player details, game results, and player season statistics.
2. **Data Fetching:** Fetches data from the MLB Stats API, including team rosters, player information, and game schedules.  The API calls are rate-limited to prevent exceeding the API's usage limits.
3. **Data Transformation:** Transforms the fetched JSON data into a format suitable for loading into BigQuery. This includes handling missing values, type conversions, and data cleaning.  It specifically converts the game date string to a datetime object.
4. **BigQuery Loading:** Loads the transformed data into the corresponding BigQuery tables using batch inserts for efficiency. It includes error handling for batch insert failures. The `update_bigquery_batch` function uses `WRITE_APPEND` to add new data, and the `real_time_updates` function uses a MERGE operation for updating games to prevent data duplication and handle potential race conditions.
5. **Real-time Updates:** Periodically fetches recent game updates, ensuring the BigQuery dataset is up-to-date. This employs a robust MERGE statement to update existing game records efficiently.

Classes:

MLBDataPipeline:
    - create_tables_if_not_exist(): Creates BigQuery dataset and tables if they don't exist.
    - fetch_players(season: int): Fetches player data for a given season.
    - fetch_teams(): Fetches data for all MLB teams.
    - load_player_season_stats(file_path: str, start_year: int, batch_size: int): Loads player season statistics from a CSV file.
    - fetch_historical_seasons(start_year: int): Fetches and loads historical MLB data.
    - process_season_games(start_date: str, end_date: str, year: int): Processes and loads games for a specific season range.
    - update_bigquery_batch(table_id: str, data: List[Dict], schema: List): Efficiently updates a BigQuery table in batches.
    - fetch_recent_games(start_date: str, end_date: str): Fetches game data within a specific date range.
    - real_time_updates(): Performs real-time updates for games, teams, and player data.

Functions:

rate_limited_request(url: str, params: Dict = None) -> requests.Response:
    - A decorator that limits the number of API requests within a specified time window.

main():
    - Entry point for the pipeline execution.


Data Structures:

team_schema, player_schema, game_schema, player_season_stats_schema:
    - BigQuery schema definitions for the respective tables.

Error Handling:

The pipeline uses `try...except` blocks to handle potential errors such as:
    - `FileNotFoundError`: For missing CSV files.
    - `requests.exceptions.RequestException`: For issues with MLB API requests.
    - `google.api_core.exceptions.GoogleAPIError`: For errors during BigQuery operations (e.g., table creation, data insertion, query execution).
    - Other `Exception` types for unexpected issues.  These are caught and logged internally for debugging.

Rate Limiting:

The pipeline uses the `ratelimit` library to implement rate limiting for MLB API requests to stay within the API usage limits.  The `CALLS` and `RATE_LIMIT` variables can be adjusted as needed.

Logging:

The `logging` module is used to provide detailed logging messages for debugging and monitoring the pipeline's execution.  Log messages include warnings and errors if any issues are encountered.

Dependencies:

google-cloud-bigquery
requests
pandas
python-dateutil
ratelimit
numpy

"""


"""
surfire.py:  MLB Data Analysis Functions

This module contains functions for analyzing MLB data using BigQuery.  Each function performs a specific analysis and returns the results in a structured format.  All functions include detailed input validation and error handling.  

Error handling includes:

* ValueError: For invalid input parameters.
* IndexError: If no data is found for a given query.
* BigQueryError (via google.api_core.exceptions):  If there's a problem with the BigQuery query execution.  This is caught and logged internally.
* Exception: For other unexpected errors during execution.


Functions:

1. get_player_highest_ops(season: int) -> Dict[str, Union[str, float]]:
   - Finds the player with the highest OPS (On-base Plus Slugging) for a given season.  Includes data validation to ensure the season is valid and there are sufficient games/at-bats played.
   - Returns: Dictionary with player name, OPS, team name.
   
2. calculate_team_home_advantage(team_name: str) -> Dict[str, Union[str, float]]:
   - Calculates a team's home field advantage (win percentage, scoring averages).  Includes data validation to ensure the team name is valid and there are enough games played to generate meaningful statistics.
   - Returns: Dictionary with home win percentage, average runs scored at home, average runs scored away, run differential, and number of wins,losses,ties.
   
3. analyze_player_performance(player_first_name: str, player_last_name: str) -> Dict[str, Union[str, float, int]]:
   - Analyzes a player's career statistics (batting average, home runs, stolen bases, games played, and at-bats).  Includes data validation to ensure player names are valid strings and that there is sufficient data to perform the analysis. 
   - Returns: Dictionary with career batting average, total home runs, total stolen bases, number of seasons played, total games played, and total at-bats.  Also includes calculated statistics like games per season and career length.
   
4. analyze_weight_performance(min_weight: int, max_weight: int) -> Dict[str, Union[float, int]]:
   - Analyzes batting performance statistics for players within a specific weight range. Includes validation to ensure that the minimum and maximum weights are valid and that the weight range is at least 10 pounds.
   - Returns: Dictionary with average batting average, average home runs, player count, minimum weight, maximum weight, and average weight.
   
5. analyze_team_strikeouts(team_name: str, season: int) -> Dict[str, Union[str, float, int]]:
   - Analyzes team strikeout statistics for a specific season. Includes validation to ensure that the team name and season are valid.
   - Returns: Dictionary with average strikeouts per game, maximum strikeouts, minimum strikeouts, total strikeouts, and games played.
   
6. analyze_monthly_home_runs(team_name: str, year: int) -> List[Dict[str, Union[str, int, float]]]:
   - Analyzes monthly home run trends for a team in a specific year. Includes validation to ensure the team name and year are valid.
   - Returns: List of dictionaries. Each dictionary contains the month, average home runs, total home runs and home runs per game for that month.
   
7. analyze_home_scoring_vs_high_away_scores(min_away_score: int = 5) -> Dict[str, float]:
   - Analyzes home team scoring when the away team scores above a certain threshold. Includes validation to ensure that the minimum away score is a valid integer.
   - Returns: Dictionary with average home score, minimum home score, maximum home score, average away score, and the number of games analyzed.
   
8. analyze_position_slugging(position: str, min_games: int = 100) -> Dict[str, Union[str, float, int]]:
   - Analyzes slugging percentage for a specific position, filtering for players with a minimum number of games played. Includes data validation to ensure that the position is valid and the number of minimum games is also valid.
   - Returns: Dictionary containing the position, average slugging percentage, number of players, minimum slugging, maximum slugging, average games played, and the number of players with slugging percentage over 0.500.
   
9. analyze_position_ops_percentile(min_games: int = 50) -> List[Dict[str, Union[str, float, int]]]:
   - Calculates the 90th percentile OPS (On-base Plus Slugging) for each position, considering players with at least a minimum number of games played. Includes data validation to ensure the minimum games is a valid number.
   - Returns: List of dictionaries containing the position, 90th percentile OPS, and the number of qualified players.
   
10. analyze_homerun_win_correlation(start_year: int, end_year: int) -> List[Dict[str, Union[int, float]]]:
    - Analyzes the correlation between team home runs and wins across multiple seasons. Includes validation to ensure start_year and end_year are valid and that the date range does not exceed 100 years.
    - Returns: List of dictionaries, where each dictionary contains a season and the correlation between home runs and wins for that season.
    
11. analyze_position_batting_efficiency(min_at_bats: int = 200) -> List[Dict[str, Union[str, int, float]]]:
    - Calculates batting efficiency metrics (average at-bats per hit) by position for players with at least a minimum number of at-bats. Includes input validation to ensure that the minimum at-bats is valid.
    - Returns: List of dictionaries, each containing position name, total at-bats, total hits, average at-bats per hit, number of players, and average batting average.
    
12. analyze_team_weight_trends(season: int) -> List[Dict[str, Union[str, int, float]]]:
    - Analyzes trends in average player weights for each team during a specific season. Includes data validation for the season.
    - Returns: A list of dictionaries, each containing team name, average player weight, minimum weight, maximum weight, and roster size.
    
13. analyze_stolen_base_efficiency(season: int) -> List[Dict[str, Union[str, int, float]]]:
    - Analyzes stolen base efficiency for each team in a given season, considering only games the team won at home. Includes input validation for the season year.
    - Returns: A list of dictionaries, each containing team name, average stolen bases per game, total stolen bases, and games played.
    
14. analyze_ops_percentile_trends() -> list:
    - Calculates the 90th percentile OPS across all seasons.
    - Returns: A list of dictionaries with the season and 90th percentile OPS for that season.
    
15. analyze_near_cycle_players(season: int, team_name: str, last_n_games: int = 5) -> List[Dict[str, Union[str, int]]]:
    - Finds players who nearly hit for the cycle (single, double, triple, home run) in a team's last N games of a season. Includes validation for the season, team name, and number of games.
    - Returns: A list of dictionaries, where each dictionary contains the first and last names of players who met the criteria.

"""





<iframe width="600" height="450" src="https://lookerstudio.google.com/embed/reporting/aafbc0bd-35cc-4f06-99c5-c1530c586bd9/page/uJ7bE" frameborder="0" style="border:0" allowfullscreen sandbox="allow-storage-access-by-user-activation allow-scripts allow-same-origin allow-popups allow-popups-to-escape-sandbox"></iframe>


Based on the schema shown, to effectively blend statistics between player_stats and games tables, we need several key pieces of data to create meaningful relationships. Here's what I recommend:

Missing Key Relationships:

- player_id (INTEGER) - Unique identifier for each player
- team_id (INTEGER) - In player_stats to match with home_team_id/away_team_id
- game_pk/game_id (INTEGER) - In player_stats to link with specific games

Copy

Insert at cursor
sql
Additional Useful Fields:

- position (STRING) - Player's primary position
- team_side (STRING) - Whether player was home/away for each game
- innings_played (INTEGER) - Track playing time
- game_status (STRING) - If player was active/inactive/injured

Copy

Insert at cursor
sql
Here's how I would structure a query to blend this data:

WITH PlayerGameStats AS (
  SELECT 
    ps.*,
    g.game_date,
    g.season,
    CASE 
      WHEN g.home_team_id = ps.team_id THEN 'HOME'
      WHEN g.away_team_id = ps.team_id THEN 'AWAY'
    END as team_side,
    g.venue_name
  FROM `mlb_data.player_stats` ps
  JOIN `mlb_data_2024.games` g 
    ON ps.game_id = g.game_id
    AND (g.home_team_id = ps.team_id OR g.away_team_id = ps.team_id)
)

Copy

Insert at cursor
sql
Key recommendations:

Add temporal tracking:

Game-by-game statistics

Player-team association dates

Season segments (pre/post All-Star break)

Add contextual data:

Weather conditions

Game time/day/night

Travel/rest days between games

Performance metrics:

Split stats (home/away, day/night)

Situational stats (runners on base, score differential)

Matchup history

Team context:

Lineup position

Defensive alignment

Platoon situations

This would allow for more comprehensive analysis like:

Player performance trends

Home/away splits

Matchup-based analytics

Team composition impact

Venue effects on performance

The current schema seems focused on individual performance metrics but lacks the relational elements needed for deeper contextual analysis.