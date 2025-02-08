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
import time
import logging
from typing import Dict, Any
import json
from datetime import datetime, timedelta


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




class AdvancedGameInfoCache:
    """
    An advanced caching mechanism for MLB team last game information
    with sophisticated invalidation and logging strategies.
    """
    def __init__(self, 
                 cache_duration: int = 3600,  # Default 1 hour
                 max_cache_size: int = 50):   # Limit cache size
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_duration = cache_duration
        self._max_cache_size = max_cache_size
        
        # Logging setup
        self._logger = logging.getLogger('GameInfoCache')
        self._logger.setLevel(logging.INFO)
        
        # Tracking cache statistics
        self._stats = {
            'hits': 0,
            'misses': 0,
            'refreshes': 0,
            'evictions': 0
        }
    
    def get(self, team: str) -> Dict[str, Any]:
        """
        Retrieve cached game information for a team if it's still valid.
        
        Args:
            team (str): The name of the MLB team
        
        Returns:
            Dict[str, Any]: Cached game information or None if not found/expired
        """
        if team in self._cache:
            cached_item = self._cache[team]
            current_time = time.time()
            
            # Check cache validity
            if current_time - cached_item['timestamp'] < self._cache_duration:
                self._stats['hits'] += 1
                self._logger.info(f"Cache HIT for team {team}")
                return cached_item['data']
            else:
                # Cache expired
                self._stats['misses'] += 1
                self._logger.info(f"Cache MISS for team {team} - Expired")
                del self._cache[team]
        
        # Cache miss
        self._stats['misses'] += 1
        self._logger.info(f"Cache MISS for team {team}")
        return None
    
    def set(self, team: str, data: Dict[str, Any]):
        """
        Store game information in the cache with intelligent management.
        
        Args:
            team (str): The name of the MLB team
            data (Dict[str, Any]): Game information to cache
        """
        # Enforce max cache size
        if len(self._cache) >= self._max_cache_size:
            # Remove oldest entry
            oldest_team = min(self._cache, key=lambda k: self._cache[k]['timestamp'])
            del self._cache[oldest_team]
            self._stats['evictions'] += 1
            self._logger.warning(f"Cache EVICTION: Removed {oldest_team} to make space")
        
        # Add new cache entry
        self._cache[team] = {
            'timestamp': time.time(),
            'data': data,
            'access_count': 0
        }
    
    def refresh(self, team: str, new_data: Dict[str, Any]):
        """
        Explicitly refresh cache entry for a team.
        
        Args:
            team (str): The name of the MLB team
            new_data (Dict[str, Any]): Updated game information
        """
        self._stats['refreshes'] += 1
        self._logger.info(f"Cache REFRESH for team {team}")
        self.set(team, new_data)
    
    def get_stats(self) -> Dict[str, int]:
        """
        Retrieve cache usage statistics.
        
        Returns:
            Dict[str, int]: Cache performance statistics
        """
        return self._stats.copy()
    
    def clear(self, team: str = None):
        """
        Clear cache for a specific team or entire cache.
        
        Args:
            team (str, optional): Team name to clear. If None, clears entire cache.
        """
        if team:
            if team in self._cache:
                del self._cache[team]
                self._logger.info(f"Manually cleared cache for {team}")
        else:
            self._cache.clear()
            self._logger.info("Entire cache cleared")

# Global cache instance
game_info_cache = AdvancedGameInfoCache()

def anchor(team: str, query_type: str = "last_game_date", force_refresh: bool = False) -> dict:
    """
    Retrieves information about the last game for a specific MLB team with advanced caching.
    
    Args:
        team (str): The name of the MLB team.
        query_type (str, optional): Type of query to perform. Defaults to "last_game_date".
        force_refresh (bool, optional): Force retrieve new data, ignoring cache. Defaults to False.
    
    Returns:
        dict: A dictionary containing information about the last game.
    """
    # Check cache first, unless force refresh is requested
    if not force_refresh:
        cached_result = game_info_cache.get(team)
        if cached_result:
            return cached_result
    
    client = genai.Client(vertexai=True, project="gem-rush-007", location="us-central1")
    MODEL_ID = "gemini-2.0-flash-exp"
    
    # Prepare the query prompt
    query_prompt = f"""
    Please provide the most recent game date for the {team} MLB team.
    
    Requirements:
    - Confirm the exact date of the last game played
    - Include the opponent team
    - Provide the game result if possible
    
    Return the response in a JSON format with the following structure:
    {{
        "team": "{team}",
        "last_game_date": "YYYY-MM-DD",
        "opponent": "Team Name",
        "result": "Win/Loss",
        "score": "Home Team Score - Away Team Score"
    }}
    """
    
    try:
        # Use Google Search tool to enhance information retrieval
        google_search_tool = Tool(google_search=GoogleSearch())
        
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=query_prompt,
            config=GenerateContentConfig(
                temperature=0.2,
                top_p=0.95,
                top_k=40,
                max_output_tokens=2048,
                tools=[google_search_tool],
                safety_settings=safety_settings,
            ),
        )
        
        # Clean and parse the response
        text = response.text
        
        # Remove code block markers if present
        if text.startswith("```"):
            text = text.split("```")[1].strip()
        if text.startswith("json"):
            text = text[4:].strip()
        
        try:
            # Parse the JSON response
            last_game_info = json.loads(text)
            
            # Cache the result (or refresh if force_refresh is True)
            if force_refresh:
                game_info_cache.refresh(team, last_game_info)
            else:
                game_info_cache.set(team, last_game_info)
            
            return last_game_info
        except json.JSONDecodeError as e:
            logging.error(f"JSON Decode Error in anchor function: {e}")
            return {
                "error": f"Failed to parse game information: {e}",
                "raw_response": text
            }
    
    except Exception as e:
        logging.error(f"Error retrieving last game information: {e}")
        return {
            "error": f"An error occurred while fetching last game details: {e}"
        }


def get_last_x_games(team: str, num_games: int) -> dict:
    """
    Retrieves information about the last X games for a specific MLB team.
    
    Args:
        team (str): The name of the MLB team
        num_games (int): Number of recent games to retrieve
    
    Returns:
        dict: Information about the last X games
    """
    client = genai.Client(vertexai=True, project="gem-rush-007", location="us-central1")
    
    query_prompt = f"""
    Please provide information about the last {num_games} games played by the {team} MLB team.
    
    Requirements:
    - List the {num_games} most recent games
    - Include date, opponent, and result for each game
    
    Return the response in a JSON format with the following structure:
    {{
        "team": "{team}",
        "games": [
            {{
                "date": "YYYY-MM-DD",
                "opponent": "Team Name",
                "result": "Win/Loss",
                "score": "Home Team Score - Away Team Score"
            }},
            ...
        ]
    }}
    """
    
    try:
        google_search_tool = Tool(google_search=GoogleSearch())
        
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=query_prompt,
            config=GenerateContentConfig(
                temperature=0.2,
                top_p=0.95,
                top_k=40,
                max_output_tokens=2048,
                tools=[google_search_tool],
                safety_settings=safety_settings,
            ),
        )
        
        text = response.text
        if text.startswith("```"):
            text = text.split("```")[1].strip()
        if text.startswith("json"):
            text = text[4:].strip()
        
        return json.loads(text)
        
    except Exception as e:
        logging.error(f"Error retrieving last {num_games} games: {e}")
        return {
            "error": f"An error occurred while fetching game details: {e}"
        }
