This test suite includes:

Unit Tests:

- Basic data fetching

- Statistics calculations

- Edge cases

- Data validation

- Format verification

- Integration Tests:

- Full pipeline testing

- Data consistency checks

- End-to-end workflow

Test Categories:

- Data Fetching

- Team Statistics

- Batting Statistics

- Pitching Statistics

- Streak Calculations

- Home/Away Splits

- Monthly Performance

- Error Handling

Fixtures:

- Sample game data

- Helper functions

- Test data generation

To run the tests:

```bash
cd mlb_fan_highlights/src/tests

# Run all tests
pytest test_historical_games.py -v

# Run specific test category
pytest test_historical_games.py -v -k "test_batting"

# Run with coverage report
pytest test_historical_games.py -v --cov=historical_games

# Run integration tests only
pytest test_historical_games.py -v -m integration
```
