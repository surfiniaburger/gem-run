import unittest
from datetime import datetime
from typing import Optional
from unittest.mock import patch

# Import your functions here (replace 'your_module' with the actual module name)
from surfire import (
    fetch_team_games,
    fetch_team_player_stats,
    fetch_team_player_stats_by_opponent,
    fetch_team_player_stats_by_game_type,
    fetch_team_plays,
    fetch_team_plays_by_opponent,
    fetch_team_plays_by_game_type,
    fetch_team_games_by_opponent,
    fetch_team_games_by_type,
    fetch_player_game_stats,
    fetch_player_plays,
    fetch_player_plays_by_opponent,
    fetch_player_plays_by_game_type,
    TEAMS
)

# Mock data for testing (replace with more comprehensive data if needed)
MOCK_GAME_RESULTS = [
    {
        'game_id': '1',
        'official_date': '2024-10-10',
        'home_team_id': 118,
        'home_team_name': 'royals',
        'away_team_id': 147,
        'away_team_name': 'yankees',
        'home_score': 5,
        'away_score': 2,
        'venue_name': 'Kauffman Stadium',
        'status': 'Final',
        'team_win': True,
        'team_margin': 3,
    }
]

MOCK_PLAYER_STATS = [
    {
        'player_id': 'player1',
        'full_name': 'Player One',
        'game_date': '2024-10-10',
        'at_bats': 4,
        'hits': 2,
        'home_runs': 1,
        'rbi': 3,
        'walks': 0,
        'strikeouts': 1,
        'batting_average': 0.5,
        'on_base_percentage': 0.5,
        'slugging_percentage': 1.25,
    }
]

MOCK_PLAYS = [
    {
        'play_id': 'play1',
        'inning': 1,
        'half_inning': 'top',
        'event': 'Home Run',
        'event_type': 'at_bat',
        'description': 'Player One hits a home run!',
        'rbi': 3,
        'is_scoring_play': True,
        'batter_name': 'Player One',
        'pitcher_name': 'Pitcher A',
        'start_time': datetime(2024, 10, 10, 19, 5, 0),
        'end_time': datetime(2024, 10, 10, 19, 6, 0),
    }
]
class TestMLBFunctions(unittest.TestCase):

    # Helper function to patch BigQuery calls and return mock data
    def _patch_bq_and_return(self, mock_data):
        def side_effect(*args, **kwargs):
            class MockQueryJob:
                def result(self):
                    return self

                def __iter__(self):
                    return iter(mock_data)

                def __getitem__(self, index):
                    return mock_data[index]

            return MockQueryJob()

        return patch('google.cloud.bigquery.Client.query', side_effect=side_effect)

    # --- Test fetch_team_games ---
    def test_fetch_team_games_success(self):
        with self._patch_bq_and_return(MOCK_GAME_RESULTS):
            games = fetch_team_games('royals', specific_date='2024-10-10')
            self.assertEqual(len(games), 1)
            self.assertEqual(games[0]['game_id'], '1')

    def test_fetch_team_games_no_results(self):
        with self._patch_bq_and_return([]):  # Empty result set
            games = fetch_team_games('royals', specific_date='2024-10-11')
            self.assertEqual(len(games), 0)

    def test_fetch_team_games_invalid_team(self):
        with self.assertRaises(KeyError):  # Expecting a KeyError for invalid team
            fetch_team_games('invalid_team')

    # --- Test fetch_team_player_stats ---
    def test_fetch_team_player_stats_success(self):
        with self._patch_bq_and_return(MOCK_PLAYER_STATS):
            stats = fetch_team_player_stats('royals', specific_date='2024-10-10')
            self.assertEqual(len(stats), 1)
            self.assertEqual(stats[0]['player_id'], 'player1')

    # --- Test fetch_team_player_stats_by_opponent ---
    def test_fetch_team_player_stats_by_opponent_success(self):
        with self._patch_bq_and_return(MOCK_PLAYER_STATS):
            stats = fetch_team_player_stats_by_opponent('royals', 'yankees', specific_date='2024-10-10')
            self.assertEqual(len(stats), 1)
            self.assertEqual(stats[0]['player_id'], 'player1')

    # --- Test fetch_team_player_stats_by_game_type ---
    def test_fetch_team_player_stats_by_game_type_success(self):
        with self._patch_bq_and_return(MOCK_PLAYER_STATS):
            stats = fetch_team_player_stats_by_game_type('royals', 'R', specific_date='2024-10-10')
            self.assertEqual(len(stats), 1)
            self.assertEqual(stats[0]['player_id'], 'player1')

    # --- Test fetch_team_plays ---
    def test_fetch_team_plays_success(self):
        with self._patch_bq_and_return(MOCK_PLAYS):
            plays = fetch_team_plays('royals', specific_date='2024-10-10')
            self.assertEqual(len(plays), 1)
            self.assertEqual(plays[0]['play_id'], 'play1')

    # --- Test fetch_team_plays_by_opponent ---
    def test_fetch_team_plays_by_opponent_success(self):
        with self._patch_bq_and_return(MOCK_PLAYS):
            plays = fetch_team_plays_by_opponent('royals', 'yankees', specific_date='2024-10-10')
            self.assertEqual(len(plays), 1)
            self.assertEqual(plays[0]['play_id'], 'play1')

    # --- Test fetch_team_plays_by_game_type ---
    def test_fetch_team_plays_by_game_type_success(self):
        with self._patch_bq_and_return(MOCK_PLAYS):
            plays = fetch_team_plays_by_game_type('royals', 'R', specific_date='2024-10-10')
            self.assertEqual(len(plays), 1)
            self.assertEqual(plays[0]['play_id'], 'play1')

    # --- Test fetch_team_games_by_opponent ---
    def test_fetch_team_games_by_opponent_success(self):
        with self._patch_bq_and_return(MOCK_GAME_RESULTS):
            games = fetch_team_games_by_opponent('royals', 'yankees', specific_date='2024-10-10')
            self.assertEqual(len(games), 1)
            self.assertEqual(games[0]['game_id'], '1')

    # --- Test fetch_team_games_by_type ---
    def test_fetch_team_games_by_type_success(self):
        with self._patch_bq_and_return(MOCK_GAME_RESULTS):
            games = fetch_team_games_by_type('royals', 'R', specific_date='2024-10-10')
            self.assertEqual(len(games), 1)
            self.assertEqual(games[0]['game_id'], '1')

    # --- Test fetch_player_game_stats ---
    def test_fetch_player_game_stats_success(self):
        with self._patch_bq_and_return(MOCK_PLAYER_STATS):
            stats = fetch_player_game_stats('royals', specific_date='2024-10-10')
            self.assertEqual(len(stats), 1)
            self.assertEqual(stats[0]['player_id'], 'player1')

    # --- Test fetch_player_plays ---
    def test_fetch_player_plays_success(self):
        with self._patch_bq_and_return(MOCK_PLAYS):
            plays = fetch_player_plays('Player One', 'royals', specific_date='2024-10-10')
            self.assertEqual(len(plays), 1)
            self.assertEqual(plays[0]['play_id'], 'play1')

    # --- Test fetch_player_plays_by_opponent ---
    def test_fetch_player_plays_by_opponent_success(self):
        with self._patch_bq_and_return(MOCK_PLAYS):
            plays = fetch_player_plays_by_opponent('Player One', 'royals', 'yankees', specific_date='2024-10-10')
            self.assertEqual(len(plays), 1)
            self.assertEqual(plays[0]['play_id'], 'play1')

    # --- Test fetch_player_plays_by_game_type ---
    def test_fetch_player_plays_by_game_type_success(self):
        with self._patch_bq_and_return(MOCK_PLAYS):
            plays = fetch_player_plays_by_game_type('Player One', 'royals', 'R', specific_date='2024-10-10')
            self.assertEqual(len(plays), 1)
            self.assertEqual(plays[0]['play_id'], 'play1')

if __name__ == '__main__':
    unittest.main()