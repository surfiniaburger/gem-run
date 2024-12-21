# test_historical_games.py
import pytest
from datetime import datetime, timedelta
from historical_games import fetch_historical_games, get_team_stats

@pytest.fixture
def sample_game_data():
    """Fixture providing sample game data for testing"""
    return [
        {
            "game_id": "123456",
            "game_date": "2024-03-28T20:10:00Z",
            "season": 2024,
            "teams": {
                "home": {
                    "team_id": 143,
                    "team_name": "Chicago Cubs",
                    "score": 5
                },
                "away": {
                    "team_id": 158,
                    "team_name": "Milwaukee Brewers",
                    "score": 2
                }
            },
            "batting_stats": {
                "home": {
                    "atBats": 34,
                    "hits": 10,
                    "doubles": 2,
                    "triples": 1,
                    "homeRuns": 1,
                    "rbi": 5,
                    "baseOnBalls": 4,
                    "strikeOuts": 8,
                    "avg": .294
                },
                "away": {
                    "atBats": 32,
                    "hits": 6,
                    "doubles": 1,
                    "triples": 0,
                    "homeRuns": 1,
                    "rbi": 2,
                    "baseOnBalls": 3,
                    "strikeOuts": 10,
                    "avg": .188
                }
            },
            "pitching_stats": {
                "home": {
                    "era": 2.00,
                    "inningsPitched": 9.0,
                    "hits": 6,
                    "runs": 2,
                    "earnedRuns": 2,
                    "baseOnBalls": 3,
                    "strikeOuts": 10,
                    "homeRuns": 1
                },
                "away": {
                    "era": 5.00,
                    "inningsPitched": 8.0,
                    "hits": 10,
                    "runs": 5,
                    "earnedRuns": 5,
                    "baseOnBalls": 4,
                    "strikeOuts": 8,
                    "homeRuns": 1
                }
            },
            "venue": "Wrigley Field",
            "status": "Final",
            "game_type": "R"
        },
        # Add more sample games as needed
    ]

class TestFetchHistoricalGames:
    def test_fetch_recent_season(self):
        """Test fetching recent season data"""
        games = fetch_historical_games(start_year=2024)
        assert len(games) > 0, "Should fetch at least some games"
        assert all(isinstance(g, dict) for g in games), "All games should be dictionaries"
    
    def test_invalid_year(self):
        """Test handling of invalid year input"""
        with pytest.raises(ValueError):
            fetch_historical_games(start_year=1800)
    
    def test_future_year(self):
        """Test handling of future year input"""
        future_year = datetime.now().year + 2
        with pytest.raises(ValueError):
            fetch_historical_games(start_year=future_year)
    
    def test_required_fields(self):
        """Test that all required fields are present in game data"""
        games = fetch_historical_games(start_year=2024)
        required_fields = {'game_id', 'game_date', 'teams', 'status'}
        
        for game in games:
            assert all(field in game for field in required_fields), \
                "All required fields should be present"
    
    def test_date_format(self):
        """Test that game dates are in correct format"""
        games = fetch_historical_games(start_year=2024)
        for game in games:
            # Verify date string can be parsed
            try:
                datetime.strptime(game['game_date'], '%Y-%m-%dT%H:%M:%SZ')
            except ValueError:
                pytest.fail("Invalid date format")

class TestGetTeamStats:
    def test_basic_stats_calculation(self, sample_game_data):
        """Test basic statistics calculation"""
        stats = get_team_stats(143, sample_game_data)  # Cubs team ID
        
        assert stats['overall']['games_played'] == 1
        assert stats['overall']['wins'] == 1
        assert stats['overall']['losses'] == 0
        assert stats['overall']['win_pct'] == 1.0
    
    def test_batting_stats(self, sample_game_data):
        """Test batting statistics calculation"""
        stats = get_team_stats(143, sample_game_data)
        
        assert stats['batting']['hits'] == 10
        assert stats['batting']['at_bats'] == 34
        assert abs(stats['batting']['avg'] - (10/34)) < 0.001
        assert stats['batting']['home_runs'] == 1
    
    def test_pitching_stats(self, sample_game_data):
        """Test pitching statistics calculation"""
        stats = get_team_stats(143, sample_game_data)
        
        assert stats['pitching']['innings_pitched'] == 9.0
        assert stats['pitching']['strikeouts'] == 10
        assert abs(stats['pitching']['era'] - 2.00) < 0.001
    
    def test_empty_games_list(self):
        """Test handling of empty games list"""
        stats = get_team_stats(143, [])
        assert "error" in stats
    
    def test_invalid_team_id(self, sample_game_data):
        """Test handling of invalid team ID"""
        stats = get_team_stats(999999, sample_game_data)
        assert "error" in stats
    
    def test_streak_calculation(self):
        """Test winning/losing streak calculation"""
        # Create sample data with known streak
        streak_games = [
            {
                "game_id": str(i),
                "game_date": (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%dT%H:%M:%SZ'),
                "teams": {
                    "home": {
                        "team_id": 143,
                        "score": 5 if i % 2 == 0 else 2
                    },
                    "away": {
                        "team_id": 158,
                        "score": 2 if i % 2 == 0 else 5
                    }
                },
                "batting_stats": {"home": {}, "away": {}},
                "pitching_stats": {"home": {}, "away": {}}
            }
            for i in range(5)
        ]
        
        stats = get_team_stats(143, streak_games)
        assert abs(stats['overall']['current_streak']) > 0
    
    def test_home_away_splits(self, sample_game_data):
        """Test home/away record calculation"""
        stats = get_team_stats(143, sample_game_data)
        
        assert stats['home']['games'] == 1
        assert stats['home']['wins'] == 1
        assert stats['away']['games'] == 0
    
    def test_last_10_games(self):
        """Test last 10 games calculation"""
        # Create 15 sample games
        many_games = [
            {
                "game_id": str(i),
                "game_date": (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%dT%H:%M:%SZ'),
                "teams": {
                    "home": {
                        "team_id": 143,
                        "score": 5
                    },
                    "away": {
                        "team_id": 158,
                        "score": 2
                    }
                },
                "batting_stats": {"home": {}, "away": {}},
                "pitching_stats": {"home": {}, "away": {}}
            }
            for i in range(15)
        ]
        
        stats = get_team_stats(143, many_games)
        assert stats['last_10']['wins'] + stats['last_10']['losses'] == 10

def test_monthly_performance(sample_game_data):
    """Test monthly performance breakdown"""
    stats = get_team_stats(143, sample_game_data)
    
    current_month = datetime.now().strftime('%Y-%m')
    assert current_month in stats['monthly_performance']
    assert 'wins' in stats['monthly_performance'][current_month]

@pytest.mark.integration
def test_full_pipeline():
    """Integration test for the full data pipeline"""
    # Fetch data
    games = fetch_historical_games(start_year=2024)
    
    # Process for a specific team
    stats = get_team_stats(143, games)
    
    # Verify results
    assert isinstance(stats, dict)
    assert 'overall' in stats
    assert 'batting' in stats
    assert 'pitching' in stats

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
