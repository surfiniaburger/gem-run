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



def anchor(team: str, query_type: str = "last_game_date") -> dict:
    """
    Retrieves information about the last game for a specific MLB team.
    
    Args:
        team (str): The name of the MLB team.
        query_type (str, optional): Type of query to perform. Defaults to "last_game_date".
    
    Returns:
        dict: A dictionary containing information about the last game.
    """
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