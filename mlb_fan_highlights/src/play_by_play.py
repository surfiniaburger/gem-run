import pandas as pd
import requests
import json
from datetime import datetime
import logging
from typing import Dict, List, Tuple
from ratelimit import limits, sleep_and_retry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CALLS = 100
RATE_LIMIT = 60

@sleep_and_retry
@limits(calls=CALLS, period=RATE_LIMIT)
def call_mlb_api(url: str) -> Dict:
    """Make a rate-limited call to the MLB API"""
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

def get_dodgers_games(season: int = 2024) -> pd.DataFrame:
    """
    Fetch all Dodgers games for the specified season
    """
    url = f'https://statsapi.mlb.com/api/v1/schedule?sportId=1&season={season}&teamId=119'
    schedule_data = call_mlb_api(url)
    
    games_list = []
    for date in schedule_data.get('dates', []):
        for game in date.get('games', []):
            game_info = {
                'game_id': game['gamePk'],
                'official_date': game['officialDate'],
                'season': season,
                'home_team_id': game['teams']['home']['team']['id'],
                'home_team_name': game['teams']['home']['team']['name'],
                'home_score': game['teams']['home'].get('score', 0),
                'away_team_id': game['teams']['away']['team']['id'],
                'away_team_name': game['teams']['away']['team']['name'],
                'away_score': game['teams']['away'].get('score', 0),
                'venue_name': game['venue']['name'],
                'status': game['status']['detailedState']
            }
            
            # Calculate Dodgers-specific fields
            is_home = game_info['home_team_id'] == 119
            dodgers_score = game_info['home_score'] if is_home else game_info['away_score']
            opponent_score = game_info['away_score'] if is_home else game_info['home_score']
            
            game_info.update({
                'dodgers_win': dodgers_score > opponent_score if game_info['status'] == 'Final' else None,
                'dodgers_margin': dodgers_score - opponent_score if game_info['status'] == 'Final' else None
            })
            
            games_list.append(game_info)
    
    return pd.DataFrame(games_list)

def process_game_plays(game_pk: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Process play-by-play data for a specific game
    Returns two dataframes: plays and player_stats
    """
    url = f'https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live'
    game_data = call_mlb_api(url)
   
    
    # Process plays
    plays_list = []
    player_stats = {'batters': {}, 'pitchers': {}}
    
    all_plays = game_data['liveData']['plays']['allPlays']
    for play in all_plays:
        play_info = {
            'play_id': f"{game_pk}_{play['about']['atBatIndex']}",
            'game_id': game_pk,
            'batter_id': play['matchup']['batter']['id'],
            'pitcher_id': play['matchup']['pitcher']['id'],
            'inning': play['about']['inning'],
            'half_inning': play['about']['halfInning'],
            'event': play['result']['event'],
            'event_type': play['result']['eventType'],
            'description': play['result']['description'],
            'balls': play['count']['balls'],
            'strikes': play['count']['strikes'],
            'outs': play['count']['outs'],
            'start_time': play['about']['startTime'],
            'end_time': play['about']['endTime'],
            'rbi': play['result'].get('rbi', 0),
            'is_scoring_play': play['about']['isScoringPlay']
        }
        plays_list.append(play_info)
        
        # Update player statistics
        batter_id = play_info['batter_id']
        pitcher_id = play_info['pitcher_id']
        
        # Initialize player stats if needed
        if batter_id not in player_stats['batters']:
            player_stats['batters'][batter_id] = {
                'player_id': batter_id,
                'game_id': game_pk,
                'at_bats': 0,
                'hits': 0,
                'singles': 0,
                'doubles': 0,
                'triples': 0,
                'home_runs': 0,
                'walks': 0,
                'strikeouts': 0,
                'rbi': 0
            }
            
        if pitcher_id not in player_stats['pitchers']:
            player_stats['pitchers'][pitcher_id] = {
                'player_id': pitcher_id,
                'game_id': game_pk,
                'batters_faced': 0,
                'strikes': 0,
                'balls': 0,
                'strikeouts': 0,
                'walks': 0,
                'hits_allowed': 0,
                'runs_allowed': 0
            }
        
        # Update batting stats
        event = play_info['event']
        if event != 'Walk':
            player_stats['batters'][batter_id]['at_bats'] += 1
        
        if event == 'Single':
            player_stats['batters'][batter_id]['hits'] += 1
            player_stats['batters'][batter_id]['singles'] += 1
        elif event == 'Double':
            player_stats['batters'][batter_id]['hits'] += 1
            player_stats['batters'][batter_id]['doubles'] += 1
        elif event == 'Triple':
            player_stats['batters'][batter_id]['hits'] += 1
            player_stats['batters'][batter_id]['triples'] += 1
        elif event == 'Home Run':
            player_stats['batters'][batter_id]['hits'] += 1
            player_stats['batters'][batter_id]['home_runs'] += 1
        elif event == 'Strikeout':
            player_stats['batters'][batter_id]['strikeouts'] += 1
        elif event == 'Walk':
            player_stats['batters'][batter_id]['walks'] += 1
        
        player_stats['batters'][batter_id]['rbi'] += play_info['rbi']
        
        # Update pitching stats
        player_stats['pitchers'][pitcher_id]['batters_faced'] += 1
        player_stats['pitchers'][pitcher_id]['strikes'] += play_info['strikes']
        player_stats['pitchers'][pitcher_id]['balls'] += play_info['balls']
        
        if event == 'Strikeout':
            player_stats['pitchers'][pitcher_id]['strikeouts'] += 1
        elif event == 'Walk':
            player_stats['pitchers'][pitcher_id]['walks'] += 1
        elif event in ['Single', 'Double', 'Triple', 'Home Run']:
            player_stats['pitchers'][pitcher_id]['hits_allowed'] += 1
        
        if play_info['is_scoring_play']:
            player_stats['pitchers'][pitcher_id]['runs_allowed'] += play_info['rbi']
    
    # Convert plays list to DataFrame
    plays_df = pd.DataFrame(plays_list)
    
    # Convert player stats to DataFrame and calculate additional metrics
    batter_stats = pd.DataFrame(list(player_stats['batters'].values()))
    pitcher_stats = pd.DataFrame(list(player_stats['pitchers'].values()))
    
    # Calculate batting averages and other metrics
    if not batter_stats.empty:
        batter_stats['batting_average'] = batter_stats['hits'] / batter_stats['at_bats']
        batter_stats['on_base_percentage'] = (batter_stats['hits'] + batter_stats['walks']) / \
                                           (batter_stats['at_bats'] + batter_stats['walks'])
        batter_stats['slugging_percentage'] = (batter_stats['singles'] + 
                                             2 * batter_stats['doubles'] + 
                                             3 * batter_stats['triples'] + 
                                             4 * batter_stats['home_runs']) / \
                                            batter_stats['at_bats']
    
    # Combine batter and pitcher stats
    player_stats_df = pd.concat([batter_stats, pitcher_stats], ignore_index=True)
    
    return plays_df, player_stats_df

def process_recent_games(n_games: int = 5) -> Dict[str, pd.DataFrame]:
    """
    Process the n most recent Dodgers games
    Returns dictionary with games, plays, and player stats DataFrames
    """
    games_df = get_dodgers_games()
    recent_games = games_df.sort_values('official_date', ascending=False).head(n_games)
    
    all_plays = []
    all_player_stats = []
    
    for game_id in recent_games['game_id']:
        try:
            plays_df, player_stats_df = process_game_plays(game_id)
            all_plays.append(plays_df)
            all_player_stats.append(player_stats_df)
        except Exception as e:
            logger.error(f"Error processing game {game_id}: {str(e)}")
    
    return {
        'games': recent_games,
        'plays': pd.concat(all_plays, ignore_index=True) if all_plays else pd.DataFrame(),
        'player_stats': pd.concat(all_player_stats, ignore_index=True) if all_player_stats else pd.DataFrame()
    }

if __name__ == "__main__":
    # Process last 5 games
    logger.info("Processing recent Dodgers games...")
    results = process_recent_games(10)
    
    # Save results to CSV files
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    for name, df in results.items():
        filename = f"dodgers_{name}_{timestamp}.csv"
        df.to_csv(filename, index=False)
        logger.info(f"Saved {name} data to {filename}")