from google.cloud import bigquery
from google.api_core import retry
import requests
import json
import pandas as pd
from datetime import datetime, timedelta, timezone, UTC
import time
from typing import List, Dict
import logging
from concurrent.futures import ThreadPoolExecutor
from ratelimit import limits, sleep_and_retry
import numpy as np


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Rate limiting for MLB API
CALLS = 100
RATE_LIMIT = 60

# Schema definitions
team_schema = [
    bigquery.SchemaField("team_id", "INTEGER"),
    bigquery.SchemaField("name", "STRING"),
    bigquery.SchemaField("team_code", "STRING"),
    bigquery.SchemaField("file_code", "STRING"),
    bigquery.SchemaField("abbreviation", "STRING"),
    bigquery.SchemaField("team_name", "STRING"),
    bigquery.SchemaField("location_name", "STRING"),
    bigquery.SchemaField("league_id", "INTEGER"),
    bigquery.SchemaField("division_id", "INTEGER"),
    bigquery.SchemaField("venue_id", "INTEGER"),
    bigquery.SchemaField("spring_venue_id", "INTEGER"),
    bigquery.SchemaField("first_year_of_play", "STRING"),
    bigquery.SchemaField("active", "BOOLEAN"),
    bigquery.SchemaField("last_updated", "TIMESTAMP")
]

player_schema = [
    bigquery.SchemaField("player_id", "INTEGER"),
    bigquery.SchemaField("full_name", "STRING"),
    bigquery.SchemaField("first_name", "STRING"),
    bigquery.SchemaField("last_name", "STRING"),
    bigquery.SchemaField("primary_number", "STRING"),
    bigquery.SchemaField("birth_date", "TIMESTAMP"),
    bigquery.SchemaField("current_team_id", "INTEGER"),
    bigquery.SchemaField("position_code", "STRING"),
    bigquery.SchemaField("position_name", "STRING"),
    bigquery.SchemaField("position_type", "STRING"),
    bigquery.SchemaField("height", "STRING"),
    bigquery.SchemaField("weight", "INTEGER"),
    bigquery.SchemaField("primary_position", "STRING"),
    bigquery.SchemaField("active", "BOOLEAN"),
    bigquery.SchemaField("last_updated", "TIMESTAMP")
]

game_schema = [
    bigquery.SchemaField("game_pk", "INTEGER", mode="REQUIRED"),
    bigquery.SchemaField("game_type", "STRING"),
    bigquery.SchemaField("season", "INTEGER"),
    bigquery.SchemaField("game_date", "TIMESTAMP"),
    bigquery.SchemaField("home_team_id", "INTEGER"),
    bigquery.SchemaField("away_team_id", "INTEGER"),
    bigquery.SchemaField("venue_id", "INTEGER"),
    bigquery.SchemaField("status", "STRING"),
    bigquery.SchemaField("home_score", "INTEGER"),
    bigquery.SchemaField("away_score", "INTEGER"),
    bigquery.SchemaField("last_updated", "TIMESTAMP")
]


# Add this to your schema definitions at the top (data enrichment)
player_season_stats_schema = [
    bigquery.SchemaField("season", "INTEGER"),
    bigquery.SchemaField("first_name", "STRING"),
    bigquery.SchemaField("last_name", "STRING"),
    bigquery.SchemaField("link", "STRING"),
    bigquery.SchemaField("position", "STRING"),
    bigquery.SchemaField("team", "STRING"),
    bigquery.SchemaField("games_played", "INTEGER"),
    bigquery.SchemaField("at_bats", "INTEGER"),
    bigquery.SchemaField("runs", "INTEGER"),
    bigquery.SchemaField("hits", "INTEGER"),
    bigquery.SchemaField("doubles", "INTEGER"),
    bigquery.SchemaField("triples", "INTEGER"),
    bigquery.SchemaField("homeruns", "INTEGER"),
    bigquery.SchemaField("rbi", "INTEGER"),
    bigquery.SchemaField("walks", "INTEGER"),
    bigquery.SchemaField("strikeouts", "INTEGER"),
    bigquery.SchemaField("stolen_bases", "INTEGER"),
    bigquery.SchemaField("caught_stealing", "INTEGER"),
    bigquery.SchemaField("batting_average", "FLOAT"),
    bigquery.SchemaField("on_base_percentage", "FLOAT"),
    bigquery.SchemaField("slugging_percentage", "FLOAT"),
    bigquery.SchemaField("on_base_plus_slugging", "FLOAT"),
    bigquery.SchemaField("last_updated", "TIMESTAMP")
]


# Add these retry parameters to your BigQuery client initialization
retry_config = retry.Retry(
    initial=1.0,  # Initial delay in seconds
    maximum=60.0,  # Maximum delay in seconds
    multiplier=2.0,  # Delay multiplier
    predicate=retry.if_transient_error,
)

@sleep_and_retry
@limits(calls=CALLS, period=RATE_LIMIT)
def rate_limited_request(url: str, params: Dict = None) -> requests.Response:
    return requests.get(url, params=params)

class MLBDataPipeline:
    def __init__(self):
        self.base_url = "https://statsapi.mlb.com/api/v1"
        self.client = bigquery.Client(
        )
        self.dataset_id = "mlb_data"
        
    def create_tables_if_not_exist(self):
        # First create dataset if it doesn't exist
        dataset_id = f"{self.client.project}.{self.dataset_id}"
        dataset = bigquery.Dataset(dataset_id)
    
        try:
            dataset = self.client.create_dataset(dataset, exists_ok=True)
            logger.info(f"Dataset {self.dataset_id} created or already exists")
        except Exception as e:
            logger.error(f"Error creating dataset: {str(e)}")
            raise
        
        # Then create tables
        dataset_ref = self.client.dataset(self.dataset_id)
    
        tables = {
            "teams": team_schema,
            "players": player_schema,
            "games": game_schema,
            "player_season_stats": player_season_stats_schema 
        }
    
        for table_name, schema in tables.items():
            table_ref = dataset_ref.table(table_name)
            try:
                self.client.get_table(table_ref)
                logger.info(f"Table {table_name} already exists")
            except Exception:
                table = bigquery.Table(table_ref, schema=schema)

                if table_name == "players":
                   table.clustering_fields = ["player_id"]
                elif table_name == "games":
                   table.clustering_fields = ["game_pk"]
                elif table_name == "teams":
                   table.clustering_fields = ["team_id"]
                   
                self.client.create_table(table)
                logger.info(f"Created table {table_name}")

    def fetch_players(self, season: int):
       """Fetch all players for a given season and store in BigQuery"""
       try:
           teams_response = rate_limited_request(f"{self.base_url}/teams", 
                                           params={"sportId": 1, "season": season})
           teams_data = teams_response.json()
        
           players_data = []
           for team in teams_data.get("teams", []):
               team_id = team.get("id")
               # Get roster for each team
               roster_response = rate_limited_request(
                   f"{self.base_url}/teams/{team_id}/roster",
                   params={"season": season, "rosterType": "fullSeason"}
                )
               roster_data = roster_response.json()
            
                # For each player on the roster, get detailed player info
               for roster_player in roster_data.get("roster", []):
                   player_id = roster_player.get("person", {}).get("id")
                   if player_id:
                       player_response = rate_limited_request(
                        f"{self.base_url}/people/{player_id}"
                       )
                       player = player_response.json().get("people", [])[0]
                    
                       player_data = {
                           "player_id": player.get("id"),
                           "full_name": player.get("fullName"),
                           "first_name": player.get("firstName"),
                           "last_name": player.get("lastName"),
                           "primary_number": player.get("primaryNumber"),
                           "birth_date": pd.to_datetime(player.get("birthDate")) if player.get("birthDate") else None,
                           "current_team_id": player.get("currentTeam", {}).get("id"),
                           "position_code": player.get("primaryPosition", {}).get("code"),
                           "position_name": player.get("primaryPosition", {}).get("name"),
                           "position_type": player.get("primaryPosition", {}).get("type"),
                           "height": player.get("height"),
                           "weight": player.get("weight"),
                           "primary_position": player.get("primaryPosition", {}).get("abbreviation"),
                           "active": player.get("active", True),
                           "last_updated": datetime.now(UTC)
                        }
                       players_data.append(player_data)
                    
                       # Add a small delay to respect rate limits
                       time.sleep(0.1)
            
           if players_data:
                  self.update_bigquery_batch("players", players_data, player_schema)
                  logger.info(f"Processed {len(players_data)} players for season {season}")
            
       except Exception as e:
            logger.error(f"Error processing players for season {season}: {str(e)}")


    def fetch_teams(self):
       """Fetch all MLB teams and store in BigQuery"""
       try:
           response = rate_limited_request(f"{self.base_url}/teams?sportId=1", 
                                     params={"sportId": 1, "activeStatus": "BOTH"})
           data = response.json()
        
           teams_data = []
           existing_teams = set()  # To track existing teams
           for team in data.get("teams", []):
                
               
               # Create a unique identifier for each team based on name and league
               unique_team_id = (team.get("name"), team.get("league", {}).get("id"))
               if unique_team_id not in existing_teams:
                       existing_teams.add(unique_team_id)
                       team_data = {
                       "team_id": team.get("id"),
                       "name": team.get("name"),
                       "team_code": team.get("teamCode"),
                       "file_code": team.get("fileCode"),
                       "abbreviation": team.get("abbreviation"),
                       "team_name": team.get("teamName"),
                       "location_name": team.get("locationName"),
                       "league_id": team.get("league", {}).get("id"),
                       "division_id": team.get("division", {}).get("id"),
                       "venue_id": team.get("venue", {}).get("id"),
                       "spring_venue_id": team.get("springVenue", {}).get("id"),
                       "first_year_of_play": team.get("firstYearOfPlay"),
                       "active": team.get("active", True),
                       "last_updated": datetime.now(UTC)
                       }
           teams_data.append(team_data)
            
           if teams_data:
               self.update_bigquery_batch("teams", teams_data, team_schema)
               logger.info(f"Processed {len(teams_data)} teams")
            
       except Exception as e:
           logger.error(f"Error processing teams: {str(e)}")



    
    def load_player_season_stats(self, file_path: str = "datasets/mlb_season_data.csv", start_year: int = 2014, batch_size: int = 1000):

        """
         Load player season statistics from CSV file into BigQuery.
    
         Args:
            file_path (str): Path to the MLB season data CSV file
        """
        try:

            # Create the player_season_stats table if it doesn't exist
            dataset_ref = self.client.dataset(self.dataset_id)
            table_ref = dataset_ref.table("player_season_stats")
        
            try:
                self.client.get_table(table_ref)
                logger.info("Player season stats table already exists")
            except Exception:
                 table = bigquery.Table(table_ref, schema=player_season_stats_schema)
                 self.client.create_table(table)
                 logger.info("Created player season stats table")

            # First read the CSV without dtype specifications
            df = pd.read_csv(file_path, low_memory=False, dtype=str)
            df = df[df['season'] >= str(start_year)]

            # Print column names to debug
            logger.info(f"CSV columns: {df.columns.tolist()}") 
            

            # Convert integer columns to nullable Int64
            integer_columns = [
            'season', 'games_played', 'at_bats', 'runs', 'hits',
            'doubles', 'triples', 'homeruns', 'rbi',
            'walks', 'strikeouts'
        ]
            for col in integer_columns:
                if col in df.columns:
                   df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')

             # Convert float columns
            float_columns = [
               'stolen_bases', 'caught_stealing', 'batting_average',
               'on_base_percentage', 'slugging_percentage', 'on_base_plus_slugging'
            ]
            for col in float_columns:
               if col in df.columns:
                  df[col] = pd.to_numeric(df[col].replace('--', np.nan), errors='coerce')
                  

             # Convert string columns
            string_columns = ['first_name', 'last_name', 'player_link', 'position', 'team']
            for col in string_columns:
                if col in df.columns:
                   df[col] = df[col].fillna('')  # Replace NaN with empty string for string columns

             # Add last_updated timestamp
            df['last_updated'] = datetime.now(UTC)

             # Load data into BigQuery
            job_config = bigquery.LoadJobConfig(
                schema=player_season_stats_schema,
                write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE
            )

            # Convert to records and ensure no NaN values remain
            records = []
            for index, row in df.iterrows():
               record = {}
               for column in df.columns:
                   value = row[column]
                   if pd.isna(value) or value == 'nan':
                       record[column] = None
                   else:
                       record[column] = value
               records.append(record)

            total_records = len(records)
        
             # Insert data in batches
            table = self.client.get_table(table_ref)
            for i in range(0, total_records, batch_size):
               batch = records[i:i + batch_size]
               try:
                  errors = self.client.insert_rows(table, batch)
        
                  if errors:
                           logger.error(f"Errors in batch {i//batch_size + 1}: {errors}")
                  else:
                           logger.info(f"Successfully loaded batch {i//batch_size + 1} ({len(batch)} records)")
                  # Add a small delay between batches
                  time.sleep(1)
               except Exception as batch_error:
                    logger.error(f"Error in batch {i//batch_size + 1}: {str(batch_error)}")
                    continue
        except FileNotFoundError:
             logger.error(f"Could not find file at {file_path}")
             raise
        except Exception as e:
             logger.error(f"Error loading player season stats: {str(e)}")
             raise


    def fetch_historical_seasons(self, start_year: int = 2024):
        # First fetch teams (only needs to be done once)
        self.fetch_teams()

        current_year = datetime.now().year
        years = range(start_year, current_year + 2)  # Include next year
        
        for year in years:
            logger.info(f"Processing season {year}")

            # Fetch players for the season
            self.fetch_players(year)
            
            # Fetch and process regular season games
            season_start = f"{year}-03-01"  # Spring training usually starts in March
            season_end = f"{year}-11-30"    # Including postseason
            
            self.process_season_games(season_start, season_end, year)
            
            # Sleep to respect rate limits
            time.sleep(2)

    def process_season_games(self, start_date: str, end_date: str, year: int):
        """Process games for an entire season in batches"""
        params = {
            "sportId": 1,
            "startDate": start_date,
            "endDate": end_date,
            "gameType": "R,F,D,L,W"  # Regular season, Wild Card, Division Series, LCS, World Series
        }
        
        try:
            response = rate_limited_request(f"{self.base_url}/schedule", params=params)
            data = response.json()
            
            games_data = []
            for date in data.get("dates", []):
                for game in date.get("games", []):
                    # Parse the game_date string to datetime
                    game_date_str = game.get("gameDate")
                    try:
                        game_date = pd.to_datetime(game_date_str)
                    except:
                        game_date = None
                    game_data = {
                        "game_pk": game.get("gamePk"),
                        "game_type": game.get("gameType"),
                        "season": int(game.get("season")) if game.get("season") is not None else None,
                        "game_date": game_date,
                        "home_team_id": game.get("teams", {}).get("home", {}).get("team", {}).get("id"),
                        "away_team_id": game.get("teams", {}).get("away", {}).get("team", {}).get("id"),
                        "venue_id": game.get("venue", {}).get("id"),
                        "status": game.get("status", {}).get("detailedState"),
                        "home_score": game.get("teams", {}).get("home", {}).get("score"),
                        "away_score": game.get("teams", {}).get("away", {}).get("score"),
                        "last_updated": datetime.now(UTC)
                    }
                    games_data.append(game_data)
            
            if games_data:
                self.update_bigquery_batch("games", games_data, game_schema)
                logger.info(f"Processed {len(games_data)} games for {year}")
                
        except Exception as e:
            logger.error(f"Error processing games for {year}: {str(e)}")
    


    def update_bigquery_batch(self, table_id: str, data: List[Dict], schema: List):
       """Update BigQuery table with batch data using MERGE operation"""
       table_ref = f"{self.client.project}.{self.dataset_id}.{table_id}"
    
       # Create a temporary table with the new data
       temp_table_id = f"{table_ref}_temp"
       job_config = bigquery.LoadJobConfig(
           schema=schema,
           write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE
       )
    
       df = pd.DataFrame(data)
       job = self.client.load_table_from_dataframe(
           df, temp_table_id, job_config=job_config
       )
       job.result()

       # Different MERGE logic based on table type
       if table_id == "players":
           merge_query = f"""
               MERGE `{table_ref}` T
               USING `{temp_table_id}` S
               ON T.player_id = S.player_id
               WHEN MATCHED THEN
                   UPDATE SET 
                       current_team_id = S.current_team_id,
                       active = S.active,
                       last_updated = S.last_updated
               WHEN NOT MATCHED THEN
                   INSERT ROW
           """
       elif table_id == "games":
           merge_query = f"""
               MERGE `{table_ref}` T
               USING `{temp_table_id}` S
               ON T.game_pk = S.game_pk
               WHEN MATCHED THEN
                   UPDATE SET 
                       status = S.status,
                       home_score = S.home_score,
                       away_score = S.away_score,
                       last_updated = S.last_updated
               WHEN NOT MATCHED THEN
                   INSERT ROW
           """
       else:
           # For other tables, just use a simple primary key merge
           merge_query = f"""
               MERGE `{table_ref}` T
               USING `{temp_table_id}` S
               ON T.{table_id[:-1]}_id = S.{table_id[:-1]}_id
               WHEN NOT MATCHED THEN
                   INSERT ROW
           """
    
       try:
           self.client.query(merge_query).result()
       finally:
           # Clean up temporary table
           self.client.delete_table(temp_table_id, not_found_ok=True)    


    
    def fetch_recent_games(self, start_date: str, end_date: str) -> List[Dict]:
       """
       Fetch games within a specific date range
    
       Args:
           start_date: Start date in YYYY-MM-DD format
           end_date: End date in YYYY-MM-DD format
    
       Returns:
           List of game dictionaries
       """
       try:
           params = {
               "sportId": 1,
               "startDate": start_date,
               "endDate": end_date,
               "gameType": "R,F,D,L,W"  # Regular season, Finals, Division, League, Wild Card
           }
        
           response = rate_limited_request(f"{self.base_url}/schedule", params=params)
           data = response.json()
        
           games_data = []
           for date in data.get("dates", []):
               for game in date.get("games", []):
                  game_date_str = game.get("gameDate")
                  try:
                      game_date = pd.to_datetime(game_date_str)
                  except:
                      game_date = None

                  game_data = {
                       "game_pk": game.get("gamePk"),
                       "game_type": game.get("gameType"),
                       "season": int(game.get("season")) if game.get("season") is not None else None,
                       "game_date": game_date,
                       "home_team_id": game.get("teams", {}).get("home", {}).get("team", {}).get("id"),
                       "away_team_id": game.get("teams", {}).get("away", {}).get("team", {}).get("id"),
                       "venue_id": game.get("venue", {}).get("id"),
                       "status": game.get("status", {}).get("detailedState"),
                       "home_score": game.get("teams", {}).get("home", {}).get("score"),
                       "away_score": game.get("teams", {}).get("away", {}).get("score"),
                       "last_updated": datetime.now(UTC)
                    }
                  games_data.append(game_data)
        
           return games_data

       except Exception as e:
           logger.error(f"Error fetching recent games: {str(e)}")
           return []
       

    
    def real_time_updates(self):
       """Perform real-time updates for games, teams, and player data"""
       try:
           current_time = datetime.now(UTC)
           logger.info(f"Starting real-time update at {current_time}")

           # 1. Update teams first (as players depend on team data)
           teams_data = self.fetch_teams()
           if teams_data:
            self.update_bigquery_batch("teams", teams_data, team_schema)

           # 2. Update players for current season
           current_year = current_time.year
           players_data = self.fetch_players(current_year)
           if players_data:
            self.update_bigquery_batch("players", players_data, player_schema)

           # 3. Update games
           # Calculate date range for recent games (e.g., last 7 days to next 7 days)
           start_date = (current_time - timedelta(days=7)).strftime('%Y-%m-%d')
           end_date = (current_time + timedelta(days=7)).strftime('%Y-%m-%d')
           games_data = self.fetch_recent_games(start_date, end_date)
           if games_data:
            self.update_bigquery_batch("games", games_data, game_schema)
            logger.info(f"Updated {len(games_data)} recent games")
        
           params = {
               "sportId": 1,
               "startDate": start_date,
               "endDate": end_date,
               "gameType": "R,F,D,L,W"  # Regular season, Finals, Division, League, Wild Card
            }

           response = rate_limited_request(f"{self.base_url}/schedule", params=params)
           data = response.json()
        
           games_data = []
           for date in data.get("dates", []):
               for game in date.get("games", []):
                   game_date_str = game.get("gameDate")
                   try:
                       game_date = pd.to_datetime(game_date_str)
                   except:
                       game_date = None

                   game_data = {
                       "game_pk": game.get("gamePk"),
                       "game_type": game.get("gameType"),
                       "season": int(game.get("season")) if game.get("season") is not None else None,
                       "game_date": game_date,
                       "home_team_id": game.get("teams", {}).get("home", {}).get("team", {}).get("id"),
                       "away_team_id": game.get("teams", {}).get("away", {}).get("team", {}).get("id"),
                       "venue_id": game.get("venue", {}).get("id"),
                       "status": game.get("status", {}).get("detailedState"),
                       "home_score": game.get("teams", {}).get("home", {}).get("score"),
                       "away_score": game.get("teams", {}).get("away", {}).get("score"),
                       "last_updated": datetime.now(UTC)
                    }
                   games_data.append(game_data)

           if games_data:
               # Update games table with MERGE/UPSERT logic
               table_id = f"{self.client.project}.{self.dataset_id}.games"
            
               # Create a temporary table with the new data
               temp_table_id = f"{table_id}_temp"
               job_config = bigquery.LoadJobConfig(
                   schema=game_schema,
                   write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE
                )
            
               df = pd.DataFrame(games_data)
               job = self.client.load_table_from_dataframe(
                   df, temp_table_id, job_config=job_config
               )
               job.result()

               # Perform MERGE operation
               merge_query = f"""
                   MERGE `{table_id}` T
                   USING `{temp_table_id}` S
                   ON T.game_pk = S.game_pk
                   WHEN MATCHED THEN
                       UPDATE SET 
                           status = S.status,
                           home_score = S.home_score,
                           away_score = S.away_score,
                           last_updated = S.last_updated
                   WHEN NOT MATCHED THEN
                       INSERT ROW
               """
            
               self.client.query(merge_query).result()
            
               # Clean up temporary table
               self.client.delete_table(temp_table_id)
            
               logger.info(f"Updated {len(games_data)} games")

           logger.info(f"Real-time update completed at {datetime.now(UTC)}")

       except Exception as e:
           logger.error(f"Error in real-time updates: {str(e)}")


def main():
    pipeline = MLBDataPipeline()
    
    # Create tables if they don't exist
    pipeline.create_tables_if_not_exist()
    
    # First, load historical data
    pipeline.fetch_historical_seasons(2024)

    # Load player season stats
    pipeline.load_player_season_stats("../../datasets/mlb_season_data.csv")
    
    # Then start real-time updates
    pipeline.real_time_updates()
    

if __name__ == "__main__":
    main()


