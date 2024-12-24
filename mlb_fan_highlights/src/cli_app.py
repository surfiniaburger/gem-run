# cli_app.py
import random
from mlb_core import MLBHighlightGenerator

def main():
    # Initialize the core functionality
    mlb_generator = MLBHighlightGenerator()
    
    # Get teams data
    teams_data = mlb_generator.get_teams_data()
    
    if teams_data is None:
        print("Failed to retrieve team data. Exiting.")
        return
    
    # Select random team
    selected_team = random.choice(teams_data['name'].tolist())
    selected_team_id = teams_data[teams_data['name'] == selected_team]['id'].iloc[0]
    
    print(f"Selected team: {selected_team} (ID: {selected_team_id})")
    
    # Get team games
    team_games = mlb_generator.get_team_games(selected_team_id)
    
    if not team_games:
        print(f"No games found for {selected_team}")
        return
    
    # Select random game
    random_game = random.choice(team_games)
    
    # Generate highlight
    highlight = mlb_generator.generate_highlight(random_game)
    print("\nGame Highlight:")
    print(highlight)

if __name__ == "__main__":
    main()
