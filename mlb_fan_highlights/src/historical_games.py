# historical_games.py
import requests
import pandas as pd
from datetime import datetime
import time
from typing import List, Dict, Any

def fetch_endpoint(url: str) -> dict:
    """Helper function to handle API requests with rate limiting"""
    response = requests.get(url)
    if response.status_code == 429:  # Rate limit hit
        time.sleep(1)  # Wait 1 second before retrying
        return fetch_endpoint(url)
    return response.json()

def process_game_data(game: Dict) -> Dict:
    """Process individual game data"""
    try:
        return {
            'game_id': game.get('gamePk'),
            'game_date': game.get('gameDate'),
            'season': game.get('season'),
            'teams': {
                'home': {
                    'team_id': game['teams']['home']['team'].get('id'),
                    'team_name': game['teams']['home']['team'].get('name'),
                    'score': game['teams']['home'].get('score', 0)
                },
                'away': {
                    'team_id': game['teams']['away']['team'].get('id'),
                    'team_name': game['teams']['away']['team'].get('name'),
                    'score': game['teams']['away'].get('score', 0)
                }
            },
            'venue': game.get('venue', {}).get('name'),
            'status': game.get('status', {}).get('detailedState'),
            'game_type': game.get('gameType'),
            'season_display': game.get('seasonDisplay')
        }
    except KeyError as e:
        print(f"Error processing game {game.get('gamePk')}: {str(e)}")
        return None

def fetch_historical_games(start_year: int = 2015) -> List[Dict[str, Any]]:
    """
    Fetch historical MLB games data from the specified start year to present.
    
    Args:
        start_year (int): The year to start fetching data from (default: 2015)
    
    Returns:
        List[Dict]: List of processed game data dictionaries
    """
    def fetch_season_games(season: int) -> List[Dict]:
        """Fetch all games for a specific season"""
        games = []
        schedule_url = f'https://statsapi.mlb.com/api/v1/schedule?sportId=1&season={season}&gameType=R'
        
        schedule_data = fetch_endpoint(schedule_url)
        if 'dates' in schedule_data:
            for date in schedule_data['dates']:
                for game in date.get('games', []):
                    processed_game = process_game_data(game)
                    if processed_game:
                        game_detail_url = f"https://statsapi.mlb.com/api/v1/game/{processed_game['game_id']}/boxscore"
                        game_details = fetch_endpoint(game_detail_url)
                        
                        processed_game['batting_stats'] = {
                            'home': game_details.get('teams', {}).get('home', {}).get('teamStats', {}).get('batting', {}),
                            'away': game_details.get('teams', {}).get('away', {}).get('teamStats', {}).get('batting', {})
                        }
                        
                        processed_game['pitching_stats'] = {
                            'home': game_details.get('teams', {}).get('home', {}).get('teamStats', {}).get('pitching', {}),
                            'away': game_details.get('teams', {}).get('away', {}).get('teamStats', {}).get('pitching', {})
                        }
                        
                        games.append(processed_game)
        return games

    def calculate_advanced_metrics(games: List[Dict]) -> List[Dict]:
        """Calculate advanced metrics for each game"""
        for game in games:
            home_score = game['teams']['home']['score']
            away_score = game['teams']['away']['score']
            game['run_differential'] = home_score - away_score
            
            try:
                home_ba = game['batting_stats']['home'].get('avg', 0)
                away_ba = game['batting_stats']['away'].get('avg', 0)
                game['batting_comparison'] = {
                    'home_ba': home_ba,
                    'away_ba': away_ba,
                    'ba_differential': float(home_ba) - float(away_ba)
                }
            except (KeyError, TypeError):
                game['batting_comparison'] = None
            
            game['home_team_result'] = 'W' if home_score > away_score else 'L'
        
        return games

    # Main execution
    current_year = datetime.now().year
    historical_games = []
    
    for year in range(start_year, current_year + 1):
        print(f"Fetching data for season {year}...")
        season_games = fetch_season_games(year)
        season_games = calculate_advanced_metrics(season_games)
        historical_games.extend(season_games)
        time.sleep(0.1)  # Rate limiting
    
    # Convert to DataFrame for additional processing
    df_games = pd.DataFrame(historical_games)
    
    # Add time-based features
    df_games['year'] = pd.to_datetime(df_games['game_date']).dt.year
    df_games['month'] = pd.to_datetime(df_games['game_date']).dt.month
    df_games['day_of_week'] = pd.to_datetime(df_games['game_date']).dt.day_name()
    
    # Calculate team rolling stats
    team_ids = set(
        df_games['teams'].apply(lambda x: x['home']['team_id']).unique().tolist() +
        df_games['teams'].apply(lambda x: x['away']['team_id']).unique().tolist()
    )
    
    for team_id in team_ids:
        team_mask = (
            (df_games['teams'].apply(lambda x: x['home']['team_id']) == team_id) |
            (df_games['teams'].apply(lambda x: x['away']['team_id']) == team_id)
        )
        team_games = df_games[team_mask].sort_values('game_date')
        
        team_games['rolling_runs'] = team_games.apply(
            lambda row: row['teams']['home']['score'] if row['teams']['home']['team_id'] == team_id 
            else row['teams']['away']['score']
        ).rolling(10, min_periods=1).mean()
        
        df_games.loc[team_games.index, f'team_{team_id}_rolling_runs'] = team_games['rolling_runs']
    
    return df_games.to_dict('records')

def get_team_stats(team_id: int, games: List[Dict]) -> Dict:
    """
    Get aggregated stats for a specific team across all provided games.
    
    Args:
        team_id (int): The MLB team ID to analyze
        games (List[Dict]): List of game data dictionaries
    
    Returns:
        Dict: Comprehensive team statistics including:
            - Overall record
            - Home/Away splits
            - Batting statistics
            - Pitching statistics
            - Winning/Losing streaks
            - Run differentials
            - Monthly performance
    """
    team_games = [g for g in games if 
                  g['teams']['home']['team_id'] == team_id or 
                  g['teams']['away']['team_id'] == team_id]
    
    if not team_games:
        return {"error": "No games found for specified team ID"}

    stats = {
        "overall": {
            "games_played": 0,
            "wins": 0,
            "losses": 0,
            "win_pct": 0.0,
            "runs_scored": 0,
            "runs_allowed": 0,
            "run_differential": 0,
            "current_streak": 0,
            "longest_win_streak": 0,
            "longest_lose_streak": 0
        },
        "home": {
            "games": 0,
            "wins": 0,
            "losses": 0,
            "win_pct": 0.0
        },
        "away": {
            "games": 0,
            "wins": 0,
            "losses": 0,
            "win_pct": 0.0
        },
        "batting": {
            "at_bats": 0,
            "hits": 0,
            "doubles": 0,
            "triples": 0,
            "home_runs": 0,
            "runs": 0,
            "rbi": 0,
            "walks": 0,
            "strikeouts": 0,
            "avg": 0.0,
            "obp": 0.0,
            "slg": 0.0,
            "ops": 0.0
        },
        "pitching": {
            "games": 0,
            "innings_pitched": 0.0,
            "hits_allowed": 0,
            "runs_allowed": 0,
            "earned_runs": 0,
            "walks": 0,
            "strikeouts": 0,
            "home_runs_allowed": 0,
            "era": 0.0,
            "whip": 0.0,
            "k_per_9": 0.0
        },
        "monthly_performance": {},
        "last_10": {
            "wins": 0,
            "losses": 0,
            "run_differential": 0
        }
    }
    
    current_streak = 0
    current_streak_type = None
    win_streaks = []
    lose_streaks = []
    
    # Process each game
    for game in sorted(team_games, key=lambda x: x['game_date']):
        # Determine if team is home or away
        is_home = game['teams']['home']['team_id'] == team_id
        team_side = 'home' if is_home else 'away'
        opp_side = 'away' if is_home else 'home'
        
        # Get team and opponent data
        team_data = game['teams'][team_side]
        opp_data = game['teams'][opp_side]
        
        # Basic game stats
        team_score = team_data['score']
        opp_score = opp_data['score']
        won_game = team_score > opp_score
        
        # Update overall stats
        stats['overall']['games_played'] += 1
        stats['overall']['runs_scored'] += team_score
        stats['overall']['runs_allowed'] += opp_score
        stats['overall']['run_differential'] += (team_score - opp_score)
        
        # Update home/away stats
        location = 'home' if is_home else 'away'
        stats[location]['games'] += 1
        if won_game:
            stats[location]['wins'] += 1
            stats['overall']['wins'] += 1
        else:
            stats[location]['losses'] += 1
            stats['overall']['losses'] += 1
            
        # Update streaks
        if won_game:
            if current_streak_type == 'W':
                current_streak += 1
            else:
                if current_streak_type == 'L':
                    lose_streaks.append(current_streak)
                current_streak = 1
                current_streak_type = 'W'
        else:
            if current_streak_type == 'L':
                current_streak += 1
            else:
                if current_streak_type == 'W':
                    win_streaks.append(current_streak)
                current_streak = 1
                current_streak_type = 'L'
        
        # Update batting stats
        batting_stats = game['batting_stats'][team_side]
        stats['batting']['at_bats'] += batting_stats.get('atBats', 0)
        stats['batting']['hits'] += batting_stats.get('hits', 0)
        stats['batting']['doubles'] += batting_stats.get('doubles', 0)
        stats['batting']['triples'] += batting_stats.get('triples', 0)
        stats['batting']['home_runs'] += batting_stats.get('homeRuns', 0)
        stats['batting']['runs'] += team_score
        stats['batting']['rbi'] += batting_stats.get('rbi', 0)
        stats['batting']['walks'] += batting_stats.get('baseOnBalls', 0)
        stats['batting']['strikeouts'] += batting_stats.get('strikeOuts', 0)
        
        # Update pitching stats
        pitching_stats = game['pitching_stats'][team_side]
        stats['pitching']['games'] += 1
        stats['pitching']['innings_pitched'] += float(pitching_stats.get('inningsPitched', 0))
        stats['pitching']['hits_allowed'] += pitching_stats.get('hits', 0)
        stats['pitching']['runs_allowed'] += pitching_stats.get('runs', 0)
        stats['pitching']['earned_runs'] += pitching_stats.get('earnedRuns', 0)
        stats['pitching']['walks'] += pitching_stats.get('baseOnBalls', 0)
        stats['pitching']['strikeouts'] += pitching_stats.get('strikeOuts', 0)
        stats['pitching']['home_runs_allowed'] += pitching_stats.get('homeRuns', 0)
        
        # Update monthly performance
        game_date = datetime.strptime(game['game_date'], '%Y-%m-%dT%H:%M:%SZ')
        month_key = f"{game_date.year}-{game_date.month:02d}"
        if month_key not in stats['monthly_performance']:
            stats['monthly_performance'][month_key] = {
                'wins': 0, 'losses': 0, 'run_differential': 0
            }
        stats['monthly_performance'][month_key]['wins'] += 1 if won_game else 0
        stats['monthly_performance'][month_key]['losses'] += 0 if won_game else 1
        stats['monthly_performance'][month_key]['run_differential'] += (team_score - opp_score)
    
    # Calculate derived statistics
    if stats['overall']['games_played'] > 0:
        # Win percentages
        stats['overall']['win_pct'] = stats['overall']['wins'] / stats['overall']['games_played']
        for location in ['home', 'away']:
            if stats[location]['games'] > 0:
                stats[location]['win_pct'] = stats[location]['wins'] / stats[location]['games']
        
        # Batting averages and rates
        if stats['batting']['at_bats'] > 0:
            stats['batting']['avg'] = stats['batting']['hits'] / stats['batting']['at_bats']
            plate_appearances = (stats['batting']['at_bats'] + stats['batting']['walks'])
            stats['batting']['obp'] = (stats['batting']['hits'] + stats['batting']['walks']) / plate_appearances
            total_bases = (stats['batting']['hits'] + 
                         stats['batting']['doubles'] + 
                         2 * stats['batting']['triples'] + 
                         3 * stats['batting']['home_runs'])
            stats['batting']['slg'] = total_bases / stats['batting']['at_bats']
            stats['batting']['ops'] = stats['batting']['obp'] + stats['batting']['slg']
        
        # Pitching rates
        if stats['pitching']['innings_pitched'] > 0:
            stats['pitching']['era'] = (9 * stats['pitching']['earned_runs'] / 
                                      stats['pitching']['innings_pitched'])
            stats['pitching']['whip'] = ((stats['pitching']['hits_allowed'] + 
                                        stats['pitching']['walks']) / 
                                       stats['pitching']['innings_pitched'])
            stats['pitching']['k_per_9'] = (9 * stats['pitching']['strikeouts'] / 
                                          stats['pitching']['innings_pitched'])
        
        # Streak records
        if win_streaks:
            stats['overall']['longest_win_streak'] = max(win_streaks)
        if lose_streaks:
            stats['overall']['longest_lose_streak'] = max(lose_streaks)
        stats['overall']['current_streak'] = current_streak * (1 if current_streak_type == 'W' else -1)
        
        # Last 10 games
        last_10 = team_games[-10:]
        stats['last_10']['wins'] = sum(1 for g in last_10 if 
            (g['teams']['home']['team_id'] == team_id and g['teams']['home']['score'] > g['teams']['away']['score']) or
            (g['teams']['away']['team_id'] == team_id and g['teams']['away']['score'] > g['teams']['home']['score']))
        stats['last_10']['losses'] = len(last_10) - stats['last_10']['wins']
        stats['last_10']['run_differential'] = sum(
            g['teams']['home']['score'] - g['teams']['away']['score'] if g['teams']['home']['team_id'] == team_id
            else g['teams']['away']['score'] - g['teams']['home']['score']
            for g in last_10
        )
    
    return stats

