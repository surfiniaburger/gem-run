from google.cloud import bigquery
from google.api_core import retry
import requests
import json
import pandas as pd
from datetime import datetime, timedelta
import time
from typing import List, Dict
import logging
from concurrent.futures import ThreadPoolExecutor
from ratelimit import limits, sleep_and_retry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Rate limiting for MLB API
CALLS = 100
RATE_LIMIT = 60

# Schema definitions
team_schema = [
    bigquery.SchemaField("team_id", "INTEGER", mode="REQUIRED"),
    bigquery.SchemaField("name", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("team_code", "STRING"),
    bigquery.SchemaField("file_code", "STRING"),
    bigquery.SchemaField("abbreviation", "STRING"),
    bigquery.SchemaField("team_name", "STRING"),
    bigquery.SchemaField("location_name", "STRING"),
    bigquery.SchemaField("first_year_of_play", "STRING"),
    bigquery.SchemaField("league_id", "INTEGER"),
    bigquery.SchemaField("division_id", "INTEGER"),
    bigquery.SchemaField("venue_id", "INTEGER"),
    bigquery.SchemaField("spring_venue_id", "INTEGER"),
    bigquery.SchemaField("last_updated", "TIMESTAMP")
]

player_schema = [
    bigquery.SchemaField("player_id", "INTEGER", mode="REQUIRED"),
    bigquery.SchemaField("full_name", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("current_team_id", "INTEGER"),
    bigquery.SchemaField("primary_position", "STRING"),
    bigquery.SchemaField("birth_date", "DATE"),
    bigquery.SchemaField("height", "STRING"),
    bigquery.SchemaField("weight", "INTEGER"),
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


@sleep_and_retry
@limits(calls=CALLS, period=RATE_LIMIT)
def rate_limited_request(url: str, params: Dict = None) -> requests.Response:
    return requests.get(url, params=params)

class MLBDataPipeline:
    def __init__(self):
        self.base_url = "https://statsapi.mlb.com/api/v1"
        self.client = bigquery.Client()
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
            "games": game_schema
        }
    
        for table_name, schema in tables.items():
            table_ref = dataset_ref.table(table_name)
            try:
                self.client.get_table(table_ref)
                logger.info(f"Table {table_name} already exists")
            except Exception:
                table = bigquery.Table(table_ref, schema=schema)
                self.client.create_table(table)
                logger.info(f"Created table {table_name}")


    def fetch_historical_seasons(self, start_year: int = 2014):
        current_year = datetime.now().year
        years = range(start_year, current_year + 2)  # Include next year
        
        for year in years:
            logger.info(f"Processing season {year}")
            
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
                    game_data = {
                        "game_pk": game.get("gamePk"),
                        "game_type": game.get("gameType"),
                        "season": game.get("season"),
                        "game_date": game.get("gameDate"),
                        "home_team_id": game.get("teams", {}).get("home", {}).get("team", {}).get("id"),
                        "away_team_id": game.get("teams", {}).get("away", {}).get("team", {}).get("id"),
                        "venue_id": game.get("venue", {}).get("id"),
                        "status": game.get("status", {}).get("detailedState"),
                        "home_score": game.get("teams", {}).get("home", {}).get("score"),
                        "away_score": game.get("teams", {}).get("away", {}).get("score"),
                        "last_updated": datetime.utcnow()
                    }
                    games_data.append(game_data)
            
            if games_data:
                self.update_bigquery_batch("games", games_data, game_schema)
                logger.info(f"Processed {len(games_data)} games for {year}")
                
        except Exception as e:
            logger.error(f"Error processing games for {year}: {str(e)}")

    def update_bigquery_batch(self, table_id: str, data: List[Dict], schema: List):
        """Update BigQuery table with batch data"""
        table_ref = f"{self.client.project}.{self.dataset_id}.{table_id}"
        
        job_config = bigquery.LoadJobConfig(
            schema=schema,
            write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
            ignore_unknown_values=True
        )
        
        df = pd.DataFrame(data)
        job = self.client.load_table_from_dataframe(df, table_ref, job_config=job_config)
        job.result()

    def real_time_updates(self):
        """Handle real-time updates for current and upcoming games"""
        while True:
            try:
                # Get games for next 7 days
                end_date = datetime.now() + timedelta(days=7)
                start_date = datetime.now() - timedelta(days=1)  # Include yesterday for any pending updates
                
                params = {
                    "sportId": 1,
                    "startDate": start_date.strftime("%Y-%m-%d"),
                    "endDate": end_date.strftime("%Y-%m-%d")
                }
                
                response = rate_limited_request(f"{self.base_url}/schedule", params=params)
                data = response.json()
                
                games_data = []
                for date in data.get("dates", []):
                    for game in date.get("games", []):
                        game_data = {
                            "game_pk": game.get("gamePk"),
                            "game_type": game.get("gameType"),
                            "season": game.get("season"),
                            "game_date": game.get("gameDate"),
                            "home_team_id": game.get("teams", {}).get("home", {}).get("team", {}).get("id"),
                            "away_team_id": game.get("teams", {}).get("away", {}).get("team", {}).get("id"),
                            "venue_id": game.get("venue", {}).get("id"),
                            "status": game.get("status", {}).get("detailedState"),
                            "home_score": game.get("teams", {}).get("home", {}).get("score"),
                            "away_score": game.get("teams", {}).get("away", {}).get("score"),
                            "last_updated": datetime.utcnow()
                        }
                        games_data.append(game_data)
                
                if games_data:
                    # Use MERGE in BigQuery to update existing games or insert new ones
                    merge_query = f"""
                    MERGE {self.dataset_id}.games T
                    USING (SELECT * FROM UNNEST(@games_data)) S
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
                    
                    job_config = bigquery.QueryJobConfig(
                        query_parameters=[
                            bigquery.ArrayQueryParameter("games_data", "STRUCT", games_data)
                        ]
                    )
                    
                    self.client.query(merge_query, job_config=job_config).result()
                    
                logger.info(f"Real-time update completed at {datetime.now()}")
                
            except Exception as e:
                logger.error(f"Error in real-time updates: {str(e)}")
            
            time.sleep(7200)  # Wait 2 hours before next update

def main():
    pipeline = MLBDataPipeline()
    
    # Create tables if they don't exist
    pipeline.create_tables_if_not_exist()
    
    # First, load historical data
    pipeline.fetch_historical_seasons(2014)
    
    # Then start real-time updates
    pipeline.real_time_updates()

if __name__ == "__main__":
    main()