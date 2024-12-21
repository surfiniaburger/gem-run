# Get team stats for a specific team
import historical_games


team_id = 143  # Example team ID
team_stats = historical_games.get_team_stats(team_id, historical_games)

# Print some key statistics
print(f"Overall Record: {team_stats['overall']['wins']}-{team_stats['overall']['losses']}")
print(f"Home Record: {team_stats['home']['wins']}-{team_stats['home']['losses']}")
print(f"Team ERA: {team_stats['pitching']['era']:.2f}")
print(f"Team Batting Average: {team_stats['batting']['avg']:.3f}")
print(f"Current Streak: {abs(team_stats['overall']['current_streak'])} {'wins' if team_stats['overall']['current_streak'] > 0 else 'losses'}")
