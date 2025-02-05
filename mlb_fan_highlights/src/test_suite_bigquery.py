import unittest
from datetime import datetime, date
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
    FULL_TEAM_NAMES  # Corrected: Use FULL_TEAM_NAMES, not TEAMS
)
from google.cloud import bigquery

# --- TEST DATASET CONFIGURATION ---
TEST_DATASET_ID = "royals_mlb_data_2024_test"  # Use a dedicated test dataset
TEST_PROJECT_ID = "gem-rush-007"  # Replace with your project ID

# --- Helper Functions ---
def create_test_dataset(client: bigquery.Client):
    """Creates the test dataset if it doesn't exist."""
    dataset_ref = bigquery.DatasetReference(TEST_PROJECT_ID, TEST_DATASET_ID)
    try:
        client.get_dataset(dataset_ref)
    except:
        dataset = bigquery.Dataset(dataset_ref)
        dataset.location = "US"  # Or your preferred location
        client.create_dataset(dataset)

def load_test_data(client: bigquery.Client):
    """Loads test data into the test dataset. Replace with your actual data."""
    # --- Example: Create and load a test games table ---
    games_schema = [  # Define your games table schema here
        bigquery.SchemaField("game_id", "INTEGER", mode="REQUIRED"),
        bigquery.SchemaField("official_date", "DATE", mode="REQUIRED"),
        bigquery.SchemaField("home_team_id", "INTEGER"),
        bigquery.SchemaField("away_team_id", "INTEGER"),
        bigquery.SchemaField("home_score", "INTEGER"),
        bigquery.SchemaField("away_score", "INTEGER"),
        bigquery.SchemaField("status", "STRING"),
        bigquery.SchemaField("royals_win", "BOOLEAN"),
        bigquery.SchemaField("royals_margin", "INTEGER"),
        bigquery.SchemaField("venue_name", "STRING"),
        bigquery.SchemaField("game_type", "STRING"),
        bigquery.SchemaField("season", "INTEGER"),
        bigquery.SchemaField("last_updated", "TIMESTAMP")

    ]
    games_table_ref = bigquery.TableReference(bigquery.DatasetReference(TEST_PROJECT_ID, TEST_DATASET_ID), "games")
    games_table = bigquery.Table(games_table_ref, schema=games_schema)

    try:  # Create table if it doesn't exist
      client.get_table(games_table)
    except:
      client.create_table(games_table)

    games_data = [
        #  Complete and consistent game
        {"game_id": 1, "official_date": date(2024, 9, 25), "home_team_id": 118, "away_team_id": 147,
         "home_score": 5, "away_score": 2, "status": "Final", "royals_win": True, "royals_margin": 3,
         "venue_name": "Test Stadium", "game_type": "R", "season": 2024, "last_updated" : datetime.now()},

        #  Game with missing plays (you'd also need to create a plays table)
        {"game_id": 2, "official_date": date(2024, 9, 26), "home_team_id": 118, "away_team_id": 140,
         "home_score": 3, "away_score": 4, "status": "Final", "royals_win": False, "royals_margin": -1,
         "venue_name": "Test Stadium", "game_type": "R", "season": 2024, "last_updated" : datetime.now()},
    ]
    if games_data:
      errors = client.insert_rows(games_table, games_data)
      if errors:
          print(f"errors are {errors}")
          raise Exception(f"Errors loading games test data: {errors}")
    # --- Example: Create and load a test plays table ---
    plays_schema = [ # Define your plays table schema
        bigquery.SchemaField("play_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("game_id", "INTEGER", mode="REQUIRED"),
        bigquery.SchemaField("start_time", "TIMESTAMP"),
        bigquery.SchemaField("event", "STRING"),
        bigquery.SchemaField("last_updated", "TIMESTAMP")
    ]
    plays_table_ref = bigquery.TableReference(bigquery.DatasetReference(TEST_PROJECT_ID, TEST_DATASET_ID), "plays")
    plays_table = bigquery.Table(plays_table_ref, schema=plays_schema)

    try:
      client.get_table(plays_table)
    except:
      client.create_table(plays_table)

    plays_data = [
        {"play_id": "1_1", "game_id": 1, "start_time": datetime(2024, 9, 25, 19, 0, 0), "event": "Single", "last_updated" : datetime.now()},
        {"play_id": "1_2", "game_id": 1, "start_time": datetime(2024, 9, 25, 19, 5, 0), "event": "Home Run", "last_updated" : datetime.now()},
        {"play_id": "2_1", "game_id": 2, "start_time": datetime(2024, 9, 27, 20, 0, 0), "event": "Double", "last_updated" : datetime.now()}, #Different Date
    ]
    if plays_data:
      errors = client.insert_rows(plays_table, plays_data)
      if errors:
          print(f"errors are {errors}")
          raise Exception(f"Errors loading play test data: {errors}")

class TestMLBFunctions(unittest.TestCase):
    # --- Setup and Teardown ---
    @classmethod
    def setUpClass(cls):
        """Runs once before all tests."""
        cls.client = bigquery.Client(project=TEST_PROJECT_ID)
        create_test_dataset(cls.client)
        load_test_data(cls.client)

    @classmethod
    def tearDownClass(cls):
        """Runs once after all tests."""
        #  Optionally delete the test dataset here, or leave it for debugging
        #  cls.client.delete_dataset(f"{TEST_PROJECT_ID}.{TEST_DATASET_ID}", delete_contents=True, not_found_ok=True)
        pass

    # --- Test fetch_team_games ---
    def test_fetch_team_games_success(self):
        games = fetch_team_games('royals', specific_date='2024-09-25')
        self.assertEqual(len(games), 1)
        self.assertEqual(games[0]['game_id'], 1)
        self.assertEqual(games[0]['official_date'], '2024-09-25')

    def test_fetch_team_games_no_results(self):
        games = fetch_team_games('royals', specific_date='2024-09-24')  # No game on this date
        self.assertEqual(len(games), 0)

    def test_fetch_team_games_invalid_team(self):
        with self.assertRaises(KeyError):
            fetch_team_games('invalid_team')
    # --- Test fetch_team_games_by_opponent ---
    def test_fetch_team_games_by_opponent_success(self):
        games = fetch_team_games_by_opponent('royals', 'New York Yankees', specific_date='2024-09-25')
        self.assertEqual(len(games), 1)
        self.assertEqual(games[0]['game_id'], 1)
        self.assertEqual(games[0]['official_date'], '2024-09-25')
    def test_fetch_team_games_by_type_success(self):
      games = fetch_team_games_by_type('royals', game_type='R', specific_date='2024-09-25')
      self.assertEqual(len(games), 1)
      self.assertEqual(games[0]['game_id'], 1)

    def test_fetch_team_games_by_type_no_result(self):
        games = fetch_team_games_by_type('royals', game_type='P', specific_date='2024-09-25')
        self.assertEqual(len(games), 0)
    def test_fetch_team_player_stats_success(self):
        with self.assertRaises(NotImplementedError):
          stats = fetch_team_player_stats('royals', specific_date='2024-10-10')
    def test_fetch_team_player_stats_by_opponent_success(self):
        with self.assertRaises(NotImplementedError):
          stats = fetch_team_player_stats_by_opponent('royals', 'yankees', specific_date='2024-10-10')
    def test_fetch_team_player_stats_by_game_type_success(self):
        with self.assertRaises(NotImplementedError):
          stats = fetch_team_player_stats_by_game_type('royals', 'R', specific_date='2024-10-10')
    def test_fetch_team_plays_success(self):
        with self.assertRaises(NotImplementedError):
          plays = fetch_team_plays('royals', specific_date='2024-10-10')
    def test_fetch_team_plays_by_opponent_success(self):
        with self.assertRaises(NotImplementedError):
          plays = fetch_team_plays_by_opponent('royals', 'yankees', specific_date='2024-10-10')
    def test_fetch_team_plays_by_game_type_success(self):
        with self.assertRaises(NotImplementedError):
          plays = fetch_team_plays_by_game_type('royals', 'R', specific_date='2024-10-10')
    def test_fetch_player_game_stats_success(self):
        with self.assertRaises(NotImplementedError):
          stats = fetch_player_game_stats('royals', specific_date='2024-10-10')

    def test_fetch_player_plays_success(self):
        with self.assertRaises(NotImplementedError):
          plays = fetch_player_plays('Player One', 'royals', specific_date='2024-10-10')

    def test_fetch_player_plays_by_opponent_success(self):
        with self.assertRaises(NotImplementedError):
          plays = fetch_player_plays_by_opponent('Player One', 'royals', 'yankees', specific_date='2024-10-10')
    def test_fetch_player_plays_by_game_type_success(self):
      with self.assertRaises(NotImplementedError):
        plays = fetch_player_plays_by_game_type('Player One', 'royals', 'R', specific_date='2024-10-10')
if __name__ == '__main__':
    unittest.main()