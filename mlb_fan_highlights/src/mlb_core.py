# mlb_core.py (core functionality module)
import requests
import json
import pandas as pd
from google import genai
from google.genai import types
from historical_games import fetch_historical_games, get_team_stats
import base64

class MLBHighlightGenerator:
    def __init__(self):
        self.teams_endpoint_url = 'https://statsapi.mlb.com/api/v1/teams?sportId=1'
        self.client = genai.Client(
            vertexai=True,
            project="gem-creation",
            location="us-central1"
        )
    
    def get_teams_data(self):
        """Fetches and returns all MLB teams data"""
        try:
            response = requests.get(self.teams_endpoint_url)
            response.raise_for_status()
            data = response.json()
            return pd.json_normalize(data.pop('teams'), sep='_')
        except Exception as e:
            print(f"Error fetching teams data: {e}")
            return None

    def get_team_games(self, team_id, year=2024):
        """Fetches games for a specific team"""
        historical_games = fetch_historical_games(start_year=year)
        if not historical_games:
            return None
        
        team_games = [game for game in historical_games 
                     if (game['teams']['home']['team_id'] == team_id or 
                         game['teams']['away']['team_id'] == team_id)]
        return team_games


    def generate_highlight(self, game_data):
        """Generates a highlight summary using Gemini 2.0, incorporating actual game details and team stats."""
        home_team = game_data['teams']['home']['team_id']
        away_team = game_data['teams']['away']['team_id']
        home_team_name = game_data['teams']['home']['team_name']
        away_team_name = game_data['teams']['away']['team_name']
        home_score = game_data['teams']['home']['score']
        away_score = game_data['teams']['away']['score']
        game_date = game_data['game_date']
        year = int(game_date.split('-')[0])

        # Fetch historical games first
        games = fetch_historical_games(start_year=year)
        
        if not games:
            print(f"No games found for year {year}")
            return

        # Get team stats for both teams
        home_team_stats = get_team_stats(home_team, games)
        away_team_stats = get_team_stats(away_team, games)

        # Check for errors in stats
        if "error" in home_team_stats or "error" in away_team_stats:
            print("Error fetching team stats")
            return

        # Create stats context string
        stats_context = f"""
Team Statistics:
{home_team_name}:
- Overall Record: {home_team_stats['overall']['wins']}-{home_team_stats['overall']['losses']}
- Win Percentage: {home_team_stats['overall']['wins']/(home_team_stats['overall']['games_played'])*100:.1f}%
- Run Differential: {home_team_stats['overall']['run_differential']}
- Home Record: {home_team_stats['home']['wins']}-{home_team_stats['home']['losses']}
- Away Record: {home_team_stats['away']['wins']}-{home_team_stats['away']['losses']}

{away_team_name}:
- Overall Record: {away_team_stats['overall']['wins']}-{away_team_stats['overall']['losses']}
- Win Percentage: {away_team_stats['overall']['wins']/(away_team_stats['overall']['games_played'])*100:.1f}%
- Run Differential: {away_team_stats['overall']['run_differential']}
- Home Record: {away_team_stats['home']['wins']}-{away_team_stats['home']['losses']}
- Away Record: {away_team_stats['away']['wins']}-{away_team_stats['away']['losses']}
"""

        prompt_text = f"""Provide a captivating highlight summary for the baseball game between the {home_team_name} and the {away_team_name} on {game_date}. The final score was {home_score} - {away_score}.

Here are the current season statistics for both teams:
{stats_context}

Please generate an engaging recap, covering:
1. Who won and the final outcome, considering the teams' current season performance.
2. Key pivotal moments in the game - including specific plays, who made them, and why they were important.
3. Notable performances by specific players, including their names and the impact they had on the game.
4. Any historical or contextual information that provides depth to the game's significance, incorporating how this game affects their season statistics.
Aim for a summary that's detailed yet concise, approximately 4-5 sentences long.
"""
        

        text_part = types.Part.from_text(prompt_text)
        
        contents = [
            types.Content(
                role="user",
                parts=[text_part]
            )
        ]

        tools = [
            types.Tool(google_search=types.GoogleSearch())
        ]

        generate_content_config = types.GenerateContentConfig(
            temperature=1,
            top_p=0.95,
            max_output_tokens=8192,
            response_modalities=["TEXT"],
            safety_settings=[
                types.SafetySetting(
                    category="HARM_CATEGORY_HATE_SPEECH",
                    threshold="OFF"
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_DANGEROUS_CONTENT",
                    threshold="OFF"
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    threshold="OFF"
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_HARASSMENT",
                    threshold="OFF"
                )
            ],
            tools=tools,
             system_instruction=[types.Part.from_text("""You are an expert MLB analyst and storyteller, akin to a blend of Vin Scully, Baseball Savant, and Baseball Reference. Your role is to:

1.  Capture the dramatic essence of baseball's defining moments with vivid descriptions and evocative language like a seasoned sports broadcaster.
2.  Incorporate relevant statistical context that enhances the storytelling and demonstrates analytical understanding of the game.
3.  Relate games to historical significance or team storylines whenever relevant to provide depth and context.
4.  Use both technical baseball language and accessible explanations to cater to a wide range of fans.
5.  Craft a narrative that resonates with the emotion and passion of baseball, from the thrill of victory to the agony of defeat, maintaining an engaging rhythm.
6.  Focus on the impact of game events on the fan base while maintaining journalistic integrity.
7.  Make sure to include player names in the summary and their significant contributions.

Your tone should blend:
- The romantic and poetic nature of baseball's oral tradition
- Modern analytical insights that are easily understood
- The gravity and importance of crucial plays and situations
- The joy, excitement and heartbreak that defines baseball as a beloved sport
- A sense of passion and enthusiasm, like a game being announced on TV or the radio.
""")], # Your existing system instruction
        )

        response_text = ""
        for chunk in self.client.models.generate_content_stream(
            model="gemini-2.0-flash-exp",
            contents=contents,
            config=generate_content_config,
        ):
            if chunk.candidates and chunk.candidates[0].content.parts:
                for part in chunk.candidates[0].content.parts:
                    if part.text:
                        response_text += part.text
        
        return response_text.strip()
