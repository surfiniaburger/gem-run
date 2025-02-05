from surfire import generate_mlb_podcasts, fetch_team_games



# Example with a date that exists in your data
try:
    games = fetch_team_games(team_name="royals", specific_date="2024-08-25")
    if games:
        for game in games:
            print(game)
    else:
        print("No games found for the specified date.")
except Exception as e:
    print(f"An error occurred: {e}")

