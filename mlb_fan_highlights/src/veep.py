from historical_games import fetch_historical_games, get_team_stats
import matplotlib.pyplot as plt

def analyze_team_performance(team_id: int, year: int = 2024):
    # Fetch the historical games data
    games = fetch_historical_games(start_year=year)
    
    if not games:
        print(f"No games found for year {year}")
        return
    
    # Get team statistics
    stats = get_team_stats(team_id, games)
    
    if "error" in stats:
        print(f"Error: {stats['error']}")
        return
    
    # Create visualization of key metrics
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Win-Loss Record
    labels = ['Wins', 'Losses']
    sizes = [stats['overall']['wins'], stats['overall']['losses']]
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%')
    ax1.set_title('Win-Loss Record')
    
    # Home vs Away Performance
    locations = ['Home', 'Away']
    home_win_pct = (stats['home']['wins'] / stats['home']['games'] * 100) if stats['home']['games'] > 0 else 0
    away_win_pct = (stats['away']['wins'] / stats['away']['games'] * 100) if stats['away']['games'] > 0 else 0
    win_pcts = [home_win_pct, away_win_pct]
    
    ax2.bar(locations, win_pcts)
    ax2.set_title('Win Percentage: Home vs Away')
    ax2.set_ylabel('Win Percentage')
    
    # Print key statistics
    print(f"\nTeam Statistics Summary:")
    print(f"Overall Record: {stats['overall']['wins']}-{stats['overall']['losses']}")
    print(f"Win Percentage: {stats['overall']['wins']/(stats['overall']['games_played'])*100:.1f}%")
    print(f"Run Differential: {stats['overall']['run_differential']}")
    print(f"Home Record: {stats['home']['wins']}-{stats['home']['losses']}")
    print(f"Away Record: {stats['away']['wins']}-{stats['away']['losses']}")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Example: Analyze New York Yankees (team_id = 147)
    # You can change the team_id and year as needed
    analyze_team_performance(team_id=147, year=2024)
