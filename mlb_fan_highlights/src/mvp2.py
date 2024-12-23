import requests
import json
import random
import pandas as pd
import google.generativeai as genai
import os
from historical_games import fetch_historical_games, get_team_stats

# ... (Existing process_endpoint_url function remains the same) ...
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



# Instead of random future game, select a random *past* game
historical_games = fetch_historical_games(start_year=2024) # Fetch only 2024 for now
if not historical_games:
    print("Failed to fetch historical games. Exiting.")
    exit()

random_game = random.choice(historical_games)  # Select a random past game

# Extract game information
home_team = random_game['teams']['home']['team_name']
away_team = random_game['teams']['away']['team_name']
game_date = random_game['game_date']
game_id = random_game['game_id']
home_score = random_game['teams']['home']['score']
away_score = random_game['teams']['away']['score']

print(f"\nSelected Game: {home_team} vs {away_team}")
print(f"Date: {game_date}")
print(f"Score: {home_team} {home_score} - {away_team} {away_score}")


# ... (Gemini setup remains the same) ...

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
model = genai.GenerativeModel(model_name="gemini-1.5-pro-002") # or another suitable model


def generate_highlight(game_data):
    """Generates a highlight summary using Gemini, incorporating actual game details."""
    home_team = game_data['teams']['home']['team_name']
    away_team = game_data['teams']['away']['team_name']
    home_score = game_data['teams']['home']['score']
    away_score = game_data['teams']['away']['score']
    game_date = game_data['game_date']

    prompt = f"""
    Generate a concise highlight summary for the game between the {home_team} and the {away_team} on {game_date}.
    The final score was {home_team} {home_score} - {away_team} {away_score}.
    
    Please create a brief, exciting summary focusing on:
    1. The final outcome
    2. Key moments that determined the game
    3. Any notable individual performances
    Keep it to 2-3 sentences.
    """
    
    response = model.generate_content([prompt])
    return response.text



# Now, generate highlight for the *single* selected game
highlight = generate_highlight(random_game)
print("\nGame Highlight:")

print(highlight)

# ... (No need for looping through team_games DataFrame)
