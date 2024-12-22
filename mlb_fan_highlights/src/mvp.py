import requests
import json
import random
import pandas as pd
import google.generativeai as genai
import os

# Function to fetch and process team data (from ds.md)
def process_endpoint_url(endpoint_url, pop_key=None):
    """Fetches data from a URL, parses JSON, and optionally pops a key."""
    try:
        response = requests.get(endpoint_url)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        data = response.json()
        if pop_key:
            df_result = pd.json_normalize(data.pop(pop_key), sep='_')
        else:
            df_result = pd.json_normalize(data)
        return df_result
    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return None
    except KeyError as e:
        print(f"KeyError: {e} not found in JSON response. Check API response structure.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None


teams_endpoint_url = 'https://statsapi.mlb.com/api/v1/teams?sportId=1'
teams_data = process_endpoint_url(teams_endpoint_url, 'teams')

if teams_data is None:
    print("Failed to retrieve team data. Exiting.")
    exit()


# Select a random team
selected_team = random.choice(teams_data['name'].tolist())
selected_team_id = teams_data[teams_data['name'] == selected_team]['id'].iloc[0]

print(f"Selected team: {selected_team} (ID: {selected_team_id})")

def get_team_schedule(team_id, season="2024"):
    schedule_endpoint_url = f'https://statsapi.mlb.com/api/v1/schedule?sportId=1&season={season}&teamId={team_id}'
    try:
        response = requests.get(schedule_endpoint_url)
        response.raise_for_status()
        data = response.json()
        
        # Check if dates exists and has content
        if not data.get('dates'):
            print("No games found in schedule")
            return None
            
        # Create empty list to store all games
        all_games = []
        
        # Process each date and its games
        for date in data['dates']:
            for game in date['games']:
                game_info = {
                    'gamePk': game['gamePk'],
                    'gameDate': game['gameDate'],
                    'home_team': game['teams']['home']['team']['name'],
                    'away_team': game['teams']['away']['team']['name']
                }
                all_games.append(game_info)
        
        # Convert to DataFrame
        if all_games:
            games_df = pd.DataFrame(all_games)
            return games_df
        else:
            print("No games found in schedule")
            return None

    except Exception as e:
        print(f"Error getting team schedule: {e}")
        return None



team_games = get_team_schedule(selected_team_id)

if team_games is None:
    print("Failed to retrieve game schedule. Exiting.")
    exit()


genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
model = genai.GenerativeModel(model_name="gemini-1.5-pro-002") # or another suitable model

def generate_highlight(game_data):
    """Generates a concise highlight summary using the Gemini API."""
    try:
        prompt = f"""Generate a concise highlight summary (one paragraph) for the {selected_team}'s game against {game_data['away_team']} on {game_data['gameDate']}.  Focus on key moments and the final score."""
        response = model.generate_content([prompt])
        return response.text
    except Exception as e:
        print(f"Error generating highlight: {e}")
        return "Error generating highlight"


team_highlights = []
for index, row in team_games.iterrows():
    game_summary = generate_highlight(row)
    team_highlights.append({"gamePk": row['gamePk'], "gameDate": row['gameDate'], "home_team":row['home_team'], "away_team":row['away_team'], "highlight": game_summary})

highlights_df = pd.DataFrame(team_highlights)
print(highlights_df.to_string())

