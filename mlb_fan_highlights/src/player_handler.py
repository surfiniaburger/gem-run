from datetime import datetime, timedelta
import streamlit as st
from typing import Dict, List, Optional
import threading
import time
from surfire2 import generate_mlb_analysis

# Cache the MLB analysis part only, since it's the expensive operation
@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_team_roster(team: str) -> List[str]:
    """
    Fetch active players for a selected team.
    
    Args:
        team: Team name
            
    Returns:
        List[str]: List of player names
    """
    try:
        # Get list of players from MLB analysis
        analysis_result = generate_mlb_analysis(f"List all current players on the {team} roster.")
        players = []
        
        # Try bullet point format first
        for line in analysis_result.split('\n'):
            if line.strip().startswith('-'):
                player_name = line.replace('-', '').strip()
                if player_name:
                    players.append(player_name)
        
        # If no players found, try comma-separated format
        if not players:
            text = analysis_result.split(':')[-1] if ':' in analysis_result else analysis_result
            text = text.replace(' and ', ', ')
            players = [name.strip() for name in text.split(',') if name.strip()]
        
        return players
        
    except Exception as e:
        st.error(f"Error fetching players: {str(e)}")
        return []

class PlayerCache:
    def __init__(self):
        self.last_refresh = datetime.now()
        self.refresh_lock = threading.Lock()
        self.cache = {}

class PlayerHandler:
    def __init__(self, headshot_handler):
        """
        Initialize PlayerHandler with HeadshotHandler instance.
        
        Args:
            headshot_handler: Instance of HeadshotHandler for managing player images
        """
        self.headshot_handler = headshot_handler
        self.player_cache = PlayerCache()
        
        # Start background refresh thread
        self.refresh_thread = threading.Thread(target=self._background_refresh, daemon=True)
        self.refresh_thread.start()
    
    def _background_refresh(self):
        """Background thread to refresh headshot URLs periodically."""
        while True:
            time.sleep(600)  # Check every 10 minutes
            current_time = datetime.now()
            
            with self.player_cache.refresh_lock:
                if (current_time - self.player_cache.last_refresh) > timedelta(minutes=20):
                    try:
                        self.headshot_handler.batch_process_headshots()
                        self.player_cache.last_refresh = current_time
                    except Exception as e:
                        print(f"Background refresh error: {str(e)}")

    def get_players_for_team(self, team: str) -> List[Dict[str, str]]:
        """
        Get players for a team using the cached roster and adding headshot info.
        
        Args:
            team: Team name
            
        Returns:
            List[Dict]: List of player information
        """
        # Get cached roster
        player_names = fetch_team_roster(team)
        players = []
        
        # Add headshot information
        for player_name in player_names:
            try:
                headshot_info = self.headshot_handler.find_player_headshot(
                    player_name=player_name,
                    team=team
                )
                
                players.append({
                    'name': player_name,
                    'team': team,
                    'headshot_url': headshot_info['signed_url'] if headshot_info else None,
                    'file_name': headshot_info['file_name'] if headshot_info else None
                })
            except Exception as e:
                print(f"Error fetching headshot for {player_name}: {str(e)}")
                players.append({
                    'name': player_name,
                    'team': team,
                    'headshot_url': None,
                    'file_name': None
                })
        
        return players
    
    def refresh_player_headshots(self, players: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Refresh headshot URLs for a list of players.
        
        Args:
            players: List of player information dictionaries
            
        Returns:
            List[Dict]: Updated player information with fresh URLs
        """
        updated_players = []
        
        for player in players:
            if player['file_name']:  # Only refresh if we have a file name
                try:
                    new_url = self.headshot_handler.get_headshot_url(player['file_name'])
                    player['headshot_url'] = new_url
                except Exception as e:
                    print(f"Error refreshing URL for {player['name']}: {str(e)}")
                    
            updated_players.append(player)
            
        return updated_players