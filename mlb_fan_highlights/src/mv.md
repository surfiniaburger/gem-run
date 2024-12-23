import requests
import json
import random
import pandas as pd
import google.generativeai as genai
import os
from historical_games import fetch_historical_games, get_team_stats
import matplotlib.pyplot as plt
from datetime import datetime

def generate_enhanced_highlight(game_data, team_stats):
    """Generates a highlight summary using Gemini, incorporating both game details and team context."""
    home_team = game_data['teams']['home']['team_name']
    away_team = game_data['teams']['away']['team_name']
    home_score = game_data['teams']['home']['score']
    away_score = game_data['teams']['away']['score']
    game_date = game_data['game_date']
    
    # Create a richer context for the highlight
    context = f"""
    Team Context:
    - {home_team} Home Record: {team_stats['home']['wins']}-{team_stats['home']['losses']}
    - Overall Run Differential: {team_stats['overall']['run_differential']}
    - Win Percentage: {team_stats['overall']['win_pct']*100:.1f}%
    """
    
    prompt = f"""
    Generate an exciting highlight summary for the game between the {home_team} and the {away_team} on {game_date}.
    The final score was {home_team} {home_score} - {away_team} {away_score}.
    
    Additional Context:
    {context}
    
    Please create a compelling summary that:
    1. Captures the game's significance in the team's season
    2. Highlights how this game affected the team's overall performance
    3. Mentions any notable statistical impacts
    Keep it to 3-4 engaging sentences.
    """
    
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    model = genai.GenerativeModel(model_name="gemini-1.5-pro-002")
    response = model.generate_content([prompt])
    return response.text

def create_game_report(team_id: int, year: int = 2024):
    """Creates a comprehensive game report with statistics and highlights."""
    # Fetch historical games
    games = fetch_historical_games(start_year=year)
    if not games:
        print("Failed to fetch historical games. Exiting.")
        return
    
    # Get team statistics
    team_stats = get_team_stats(team_id, games)
    if "error" in team_stats:
        print(f"Error: {team_stats['error']}")
        return
    
    # Filter games for selected team
    team_games = [g for g in games if 
                  g['teams']['home']['team_id'] == team_id or 
                  g['teams']['away']['team_id'] == team_id]
    
    if not team_games:
        print("No games found for selected team.")
        return
    
    # Select most recent game
    latest_game = max(team_games, key=lambda x: datetime.strptime(x['game_date'], '%Y-%m-%d'))
    
    # Generate visualizations
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # Win-Loss Record Pie Chart
    labels = ['Wins', 'Losses']
    sizes = [team_stats['overall']['wins'], team_stats['overall']['losses']]
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%', colors=['#2ecc71', '#e74c3c'])
    ax1.set_title('Season Record')
    
    # Run Differential Over Time
    games_played = range(1, team_stats['overall']['games_played'] + 1)
    cumulative_differential = []
    running_diff = 0
    
    for game in sorted(team_games, key=lambda x: x['game_date']):
        is_home = game['teams']['home']['team_id'] == team_id
        team_side = 'home' if is_home else 'away'
        opp_side = 'away' if is_home else 'home'
        running_diff += (game['teams'][team_side]['score'] - game['teams'][opp_side]['score'])
        cumulative_differential.append(running_diff)
    
    ax2.plot(games_played, cumulative_differential, color='#3498db', linewidth=2)
    ax2.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    ax2.set_title('Cumulative Run Differential Over Season')
    ax2.set_xlabel('Games Played')
    ax2.set_ylabel('Run Differential')
    
    plt.tight_layout()
    
    # Generate enhanced highlight
    highlight = generate_enhanced_highlight(latest_game, team_stats)
    
    # Print comprehensive report
    print("\n=== MLB Fan Experience Report ===")
    print(f"\nLatest Game Analysis:")
    print(f"Date: {latest_game['game_date']}")
    print(f"{latest_game['teams']['away']['team_name']} @ {latest_game['teams']['home']['team_name']}")
    print(f"Score: {latest_game['teams']['away']['score']} - {latest_game['teams']['home']['score']}")
    
    print("\nSeason Statistics:")
    print(f"Overall Record: {team_stats['overall']['wins']}-{team_stats['overall']['losses']}")
    print(f"Win Percentage: {team_stats['overall']['win_pct']*100:.1f}%")
    print(f"Run Differential: {team_stats['overall']['run_differential']}")
    
    print("\nAI-Generated Game Highlight:")
    print(highlight)
    
    plt.show()

if __name__ == "__main__":
    # Example: Create report for New York Yankees (team_id = 147)
    create_game_report(team_id=147)

Copy

Insert at cursor
python
This enhanced version includes several clever additions:

Enhanced Context : The highlight generator now incorporates team statistics to create more meaningful and contextual game summaries.

Visual Analytics :

A clean pie chart showing win-loss record

A run differential trend line showing team performance over time

Comprehensive Report Format :

Latest game details

Season statistics

AI-generated highlight with statistical context

Visual representations of team performance

Time-Based Analysis : Uses the most recent game instead of a random one, making the report more relevant

To use this enhanced version:

Make sure you have all required dependencies:

poetry add matplotlib pandas google-generativeai

Copy

Insert at cursor
bash
Set your Google API key in your environment variables:

export GOOGLE_API_KEY='your-api-key'

Copy

Insert at cursor
bash
Run the script:

poetry run python mlb_fan_highlights/src/mvp2.py

Copy

Insert at cursor
bash
This version creates a more engaging and comprehensive fan experience by combining statistical analysis with AI-generated narratives, making it more valuable for fans who want both data-driven insights and engaging storytelling about their team's performance.`