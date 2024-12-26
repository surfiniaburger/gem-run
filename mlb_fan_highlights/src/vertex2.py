from google.cloud import bigquery
from datetime import datetime
import pandas as pd
from typing import Dict, List, Optional
import aiohttp
import json




class MLBDataWarehouse:
    def __init__(self):
        self.client = bigquery.Client()
        self.dataset_id = "mlb_stats"
        self.historical_years = 10

    async def initialize_historical_data(self):
        """
        Initial load of historical MLB data into BigQuery
        Only needs to be run once or periodically for updates
        """
        try:
            current_year = datetime.now().year
            start_year = current_year - self.historical_years
            
            # Create tables if they don't exist
            await self._create_tables()
            
            for year in range(start_year, current_year + 1):
                # Fetch and store historical data
                # This would be done in batches to handle API limits
                historical_data = await self._fetch_historical_data(year)
                await self.store_data(historical_data)
        except Exception as e:
            raise Exception(f"Error Initializing BigQuery: {str(e)}")
        

    async def _create_tables(self):
        """
        Create BigQuery table schema
        """
        schema = [
            bigquery.SchemaField("team_id", "STRING"),
            bigquery.SchemaField("season", "INTEGER"),
            bigquery.SchemaField("stat_type", "STRING"),
            bigquery.SchemaField("stats", "RECORD", mode="REPEATED", fields=[
                bigquery.SchemaField("category", "STRING"),
                bigquery.SchemaField("value", "FLOAT64")
            ]),
            bigquery.SchemaField("last_updated", "TIMESTAMP"),
            bigquery.SchemaField("is_current", "BOOLEAN")
        ]

        table_id = f"{self.client.project}.{self.dataset_id}.mlb_historical_stats"
        table = bigquery.Table(table_id, schema=schema)
        table = self.client.create_table(table, exists_ok=True)

    async def get_stats(self, team_id: str, season: int, stat_type: str) -> Dict:
        """
        Retrieve stats from BigQuery first, fall back to API if needed
        """
        query = f"""
        SELECT *
        FROM `{self.dataset_id}.mlb_historical_stats`
        WHERE team_id = @team_id 
        AND season = @season 
        AND stat_type = @stat_type
        AND is_current = TRUE
        """

        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("team_id", "STRING", team_id),
                bigquery.ScalarQueryParameter("season", "INTEGER", season),
                bigquery.ScalarQueryParameter("stat_type", "STRING", stat_type)
            ]
        )

        query_job = self.client.query(query, job_config=job_config)
        results = query_job.result()

        # If data exists and is current, return it
        if results.total_rows > 0:
            return self._format_query_results(results)
        
        # If data doesn't exist or is outdated, fetch from API
        return await self._fetch_and_store_new_data(team_id, season, stat_type)

    async def _fetch_and_store_new_data(self, team_id: str, season: int, stat_type: str) -> Dict:
        """
        Fetch new data from MLB API and store in BigQuery
        """
        # Fetch from MLB API (implement API call here)
        new_data = await self._fetch_from_mlb_api(team_id, season, stat_type)
        
        # Store in BigQuery
        await self.store_data(new_data)
        
        return new_data

    async def store_data(self, data: Dict) -> bool:
        """
        Store or update data in BigQuery
        """
        try:
            # Mark existing data as not current
            update_query = f"""
            UPDATE `{self.dataset_id}.mlb_historical_stats`
            SET is_current = FALSE
            WHERE team_id = @team_id 
            AND season = @season 
            AND stat_type = @stat_type
            """
            
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("team_id", "STRING", data['team_id']),
                    bigquery.ScalarQueryParameter("season", "INTEGER", data['season']),
                    bigquery.ScalarQueryParameter("stat_type", "STRING", data['stat_type'])
                ]
            )
            
            self.client.query(update_query, job_config=job_config).result()

            # Insert new data
            table_id = f"{self.client.project}.{self.dataset_id}.mlb_historical_stats"
            rows_to_insert = [{
                'team_id': data['team_id'],
                'season': data['season'],
                'stat_type': data['stat_type'],
                'stats': data['stats'],
                'last_updated': datetime.now(),
                'is_current': True
            }]

            errors = self.client.insert_rows_json(table_id, rows_to_insert)
            return len(errors) == 0
        except Exception as e:
            raise Exception(f"Error storing data: {str(e)}")
        
        
from vertexai.generative_models import (
    GenerativeModel,
    FunctionDeclaration,
    Tool,
    GenerationConfig,
    Content
)

class MLBApiClient:
    def __init__(self):
        self.base_url = "https://statsapi.mlb.com/api/v1"
        self.session = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def _make_request(self, endpoint: str, params: Dict = None) -> Dict:
        """Make authenticated request to MLB API"""
        if not self.session:
            self.session = aiohttp.ClientSession()
            
        async with self.session.get(f"{self.base_url}/{endpoint}", params=params) as response:
            response.raise_for_status()
            return await response.json()

class MLBQueryParser:
    def __init__(self):
        self.model = GenerativeModel("gemini-1.5-pro")
        self.mlb_functions = [
            FunctionDeclaration(
                name="parse_mlb_query",
                description="Parse natural language query for MLB statistics",
                parameters={
                    "type": "object",
                    "properties": {
                        "team_id": {
                            "type": "string",
                            "description": "MLB team identifier"
                        },
                        "season": {
                            "type": "integer",
                            "description": "Season year"
                        },
                        "stat_type": {
                            "type": "string",
                            "enum": ["batting", "pitching", "fielding"],
                            "description": "Type of statistics requested"
                        },
                        "timeframe": {
                            "type": "string",
                            "enum": ["season", "game", "career"],
                            "description": "Timeframe for the statistics"
                        }
                    },
                    "required": ["team_id", "season", "stat_type"]
                }
            )
        ]
        self.tools = Tool(function_declarations=self.mlb_functions)

    async def parse_query(self, query: str) -> Dict:
        """
        Parse natural language query using Vertex AI
        """
        try:
            generation_config = GenerationConfig(
                temperature=0.2,
                top_p=0.8,
                top_k=40
            )

            # Prepare prompt with specific instructions
            prompt = f"""
            Analyze the following query and extract MLB team, season, and statistic type information.
            If specific details are missing, use reasonable defaults (current season, all stats).
            Query: {query}
            """

            response = self.model.generate_content(
                prompt,
                generation_config=generation_config,
                tools=[self.tools]
            )

            # Extract function call parameters from response
            if hasattr(response, 'candidates') and response.candidates:
                function_call = response.candidates[0].content.parts[0]
                return json.loads(function_call)
            
            raise ValueError("No valid response from model")

        except Exception as e:
            raise Exception(f"Error parsing query: {str(e)}")

class MLBDataFetcher:
    def __init__(self):
        self.api_client = MLBApiClient()
        self.team_mapping = self._load_team_mapping()

    def _load_team_mapping(self) -> Dict:
        """
        Load MLB team name to team_id mapping
        """
        return {
            "Yankees": "147",
            "Red Sox": "111",
            "Blue Jays": "141",
            "Rays": "139",
            "Orioles": "110",
            # Add more teams as needed
        }

    async def _fetch_from_mlb_api(self, team_id: str, season: int, stat_type: str) -> Dict:
        """
        Fetch data from MLB API with error handling and rate limiting
        """
        try:
            async with self.api_client as client:
                # Construct endpoint based on stat type
                if stat_type == "batting":
                    endpoint = f"teams/{team_id}/stats/batting"
                elif stat_type == "pitching":
                    endpoint = f"teams/{team_id}/stats/pitching"
                else:
                    endpoint = f"teams/{team_id}/stats/fielding"

                params = {
                    "season": season,
                    "group": "hitting" if stat_type == "batting" else stat_type,
                    "gameType": "R"  # Regular season games
                }

                response = await client._make_request(endpoint, params)

                # Transform API response to our schema
                return {
                    "team_id": team_id,
                    "season": season,
                    "stat_type": stat_type,
                    "stats": self._transform_stats(response.get("stats", [])),
                    "last_updated": datetime.now().isoformat(),
                    "is_current": True
                }

        except aiohttp.ClientError as e:
            raise Exception(f"MLB API error: {str(e)}")
        except Exception as e:
            raise Exception(f"Error fetching MLB data: {str(e)}")

    def _transform_stats(self, raw_stats: List) -> List[Dict]:
        """
        Transform raw MLB API stats to our schema
        """
        transformed_stats = []
        
        for stat_group in raw_stats:
            for stat_name, stat_value in stat_group.get("stats", {}).items():
                transformed_stats.append({
                    "category": stat_name,
                    "value": float(stat_value) if isinstance(stat_value, (int, float)) else 0.0
                })
        
        return transformed_stats

class MLBQueryHandler:
    def __init__(self):
        self.parser = MLBQueryParser()
        self.fetcher = MLBDataFetcher()
        self.warehouse = MLBDataWarehouse()  # From previous implementation

    async def process_query(self, query: str) -> Dict:
        """
        Complete process of handling MLB queries
        """
        try:
            # Parse the natural language query
            parsed_query = await self.parser.parse_query(query)
            
            # Check if data exists in warehouse
            warehouse_data = await self.warehouse.get_stats(
                team_id=parsed_query['team_id'],
                season=parsed_query['season'],
                stat_type=parsed_query['stat_type']
            )

            if warehouse_data:
                return {
                    "status": "success",
                    "source": "warehouse",
                    "data": warehouse_data,
                    "query_details": parsed_query
                }

            # If not in warehouse, fetch from API
            api_data = await self.fetcher._fetch_from_mlb_api(
                team_id=parsed_query['team_id'],
                season=parsed_query['season'],
                stat_type=parsed_query['stat_type']
            )

            # Store in warehouse for future use
            await self.warehouse.store_data(api_data)

            return {
                "status": "success",
                "source": "api",
                "data": api_data,
                "query_details": parsed_query
            }

        except Exception as e:
            return {
                "status": "error",
                "message": str(e),
                "query": query
            }


# Usage example
async def main():
    # Initialize warehouse and load historical data (run once)
    warehouse = MLBDataWarehouse()
    await warehouse.initialize_historical_data()
    
    # Create query handler
    handler = MLBQueryHandler()
    
    # Process queries
    query = "Get batting statistics for the New York Yankees in the 2023 season"
    result = await handler.process_query(query)
    print(result)


