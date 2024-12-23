# historical_games.py
import requests
import pandas as pd
from datetime import datetime
import time
from typing import List, Dict, Any
from functools import lru_cache
import os
import json

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

@lru_cache(maxsize=32)   
def fetch_historical_games(start_year=2024, end_year=None):
    """Fetch historical games data for MLB teams."""
    if end_year is None:
        end_year = start_year

    cache_file = f'games_cache_{start_year}.json'
    
    # Try to load from cache first
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Cache read error: {e}")
    
    
    all_games = []
    print(f"Fetching data for season {start_year}...")
    
    try:
        # Add rate limiting
        time.sleep(0.5)  # 500ms delay between requests
        # Get schedule for all teams
        schedule_url = f'https://statsapi.mlb.com/api/v1/schedule?sportId=1&season={start_year}'
        response = requests.get(schedule_url)
        response.raise_for_status()
        schedule_data = response.json()
        
        if 'dates' in schedule_data:
            for date in schedule_data['dates']:
                for game in date['games']:
                    # Only include completed games
                    if game['status']['detailedState'] == 'Final':
                        game_info = {
                            'game_id': game['gamePk'],
                            'game_date': game['gameDate'],
                            'teams': {
                                'home': {
                                    'team_id': game['teams']['home']['team']['id'],
                                    'team_name': game['teams']['home']['team']['name'],
                                    'score': game['teams']['home'].get('score', 0)
                                },
                                'away': {
                                    'team_id': game['teams']['away']['team']['id'],
                                    'team_name': game['teams']['away']['team']['name'],
                                    'score': game['teams']['away'].get('score', 0)
                                }
                            },
                            'status': game['status']['detailedState']
                        }
                        all_games.append(game_info)
        
        # Cache the results
        if all_games:
            try:
                with open(cache_file, 'w') as f:
                    json.dump(all_games, f)
            except Exception as e:
                print(f"Cache write error: {e}")
        
            
            
        return all_games
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

# Add cleanup function
def cleanup_resources():
    """Clean up cached resources and memory."""
    fetch_historical_games.cache_clear()  # Clear the LRU cache


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
            "run_differential": 0
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
        }
    }
    
    
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
            
        # Calculate win percentages
    if stats['overall']['games_played'] > 0:
        stats['overall']['win_pct'] = stats['overall']['wins'] / stats['overall']['games_played']
    
    if stats['home']['games'] > 0:
        stats['home']['win_pct'] = stats['home']['wins'] / stats['home']['games']
    
    if stats['away']['games'] > 0:
        stats['away']['win_pct'] = stats['away']['wins'] / stats['away']['games']

    return stats

