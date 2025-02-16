PROJECT_ID = "gem-rush-007"  # YOUR PROJECT ID
LOCATION = "us-central1"
STAGING_BUCKET = "gs://gem-rush-007-reasoning-engine"  # YOUR BUCKET

import vertexai
vertexai.init(project=PROJECT_ID, location=LOCATION, staging_bucket=STAGING_BUCKET)

# --- Install Dependencies (as in the template) ---
# %pip install --upgrade --quiet \
#     "google-cloud-aiplatform[langchain,reasoningengine]" \
#     cloudpickle==3.0.0 \
#     "pydantic>=2.10" \
#     requests \
#     google-cloud-bigquery \
#     google-cloud-secret-manager \
#     google-cloud-logging\
#     pymongo


# --- Define Model ---
model = "gemini-2.0-pro-exp-02-05"  # Start with 1.5-pro for reliability

# --- Define Tools (adapted functions from mlb_data_functions.py) ---
# IMPORTANT:  These are the *adapted* versions, as described above.

from vertexai.preview import reasoning_engines

def fetch_team_games(team_name: str, limit: int = 2, specific_date: str = None) -> list:
    """
    Fetches the most recent games (limited by 'limit') for a specified team
    using plays data to determine game recency, with optional filtering by a specific date.

    Args:
        team_name: The team name (e.g., 'rangers', 'yankees').
        limit: The maximum number of games to return.
        specific_date: A specific date in 'YYYY-MM-DD' format.

    Returns:
        A list of dictionaries, each representing a game, or an error object if an issue occurs.
    """
    from google.cloud import bigquery
    from google.cloud.bigquery import exceptions
    import logging

    logging.basicConfig(level=logging.INFO)  # Basic logging setup

    TEAMS = {
        'rangers': 140,
        'angels': 108,
        'astros': 117,
        'rays': 139,
        'blue_jays': 141,
        'yankees': 147,
        'orioles': 110,
        'red_sox': 111,
        'twins': 142,
        'white_sox': 145,
        'guardians': 114,
        'tigers': 116,
        'royals': 118,
        'padres': 135,
        'giants': 137,
        'diamondbacks': 109,
        'rockies': 115,
        'phillies': 143,
        'braves': 144,
        'marlins': 146,
        'nationals': 120,
        'mets': 121,
        'pirates': 134,
        'cardinals': 138,
        'brewers': 158,
        'cubs': 112,
        'reds': 113,
        'athletics': 133,
        'mariners': 136,
        'dodgers': 119,
    }

    FULL_TEAM_NAMES = {
        'texas rangers': 'rangers',
        'los angeles angels': 'angels',
        'houston astros': 'astros',
        'tampa bay rays': 'rays',
        'toronto blue jays': 'blue_jays',
        'new york yankees': 'yankees',
        'baltimore orioles': 'orioles',
        'boston red sox': 'red_sox',
        'minnesota twins': 'twins',
        'chicago white sox': 'white_sox',
        'cleveland guardians': 'guardians',
        'detroit tigers': 'tigers',
        'kansas city royals': 'royals',
        'san diego padres': 'padres',
        'san francisco giants': 'giants',
        'arizona diamondbacks': 'diamondbacks',
        'colorado rockies': 'rockies',
        'philadelphia phillies': 'phillies',
        'atlanta braves': 'braves',
        'miami marlins': 'marlins',
        'washington nationals': 'nationals',
        'new york mets': 'mets',
        'pittsburgh pirates': 'pirates',
        'st louis cardinals': 'cardinals',
        'milwaukee brewers': 'brewers',
        'chicago cubs': 'cubs',
        'cincinnati reds': 'reds',
        'oakland athletics': 'athletics',
        'seattle mariners': 'mariners',
        'los angeles dodgers': 'dodgers',
    }

    def _get_table_name(team_name: str) -> str:
        """Helper function to construct the table name."""
        cleaned_name = team_name.lower().strip()
        if cleaned_name in FULL_TEAM_NAMES:
            team_key = FULL_TEAM_NAMES[cleaned_name]
            return f"`gem-rush-007.{team_key}_mlb_data_2024`"
        for team_key in TEAMS:
            if team_key in cleaned_name:
                return f"`gem-rush-007.{team_key}_mlb_data_2024`"
        return f"`gem-rush-007.unknown_team_mlb_data_2024`"

    def process_game_results(query_job, specific_date=None):
        """Helper function to process query results and check against specific_date."""
        results = []
        for row in query_job:
            row_dict = dict(row)
            if row_dict.get('official_date') and not isinstance(row_dict['official_date'], str):
                row_dict['official_date'] = row_dict['official_date'].strftime('%Y-%m-%d')

            if specific_date and row_dict.get('official_date') != specific_date:
                continue

            required_fields = ['game_id', 'official_date']
            if not all(row_dict.get(field) for field in required_fields):
                continue

            numeric_fields = ['home_score', 'away_score', 'team_margin']
            for field in numeric_fields:
                try:
                    if row_dict.get(field) is not None:
                        row_dict[field] = int(row_dict[field])
                except (TypeError, ValueError):
                    row_dict[field] = None
            results.append(row_dict)

        return results

    team_id = TEAMS[team_name]
    table_name = _get_table_name(team_name)

    try:
        query = f"""
            SELECT
                g.game_id,
                g.official_date,
                g.home_team_id,
                g.home_team_name,
                g.away_team_id,
                g.away_team_name,
                g.home_score,
                g.away_score,
                g.venue_name,
                g.status,
                {team_name}_win as team_win,
                {team_name}_margin as team_margin,
                subquery.max_end_time
            FROM
                {table_name}.games AS g
            INNER JOIN
                (SELECT
                    game_id,
                    MAX(end_time) AS max_end_time
                FROM
                    {table_name}.plays
        """

        if specific_date:
            query += f" WHERE DATE(end_time) = @specific_date"

        query += f"""
                GROUP BY game_id
                ) AS subquery
                ON g.game_id = subquery.game_id
            WHERE
                (g.home_team_id = {team_id} OR g.away_team_id = {team_id})
        """

        if specific_date:
            query += f" AND g.official_date = @specific_date"

        query += " ORDER BY subquery.max_end_time DESC LIMIT @limit"

        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("limit", "INT64", limit),
                bigquery.ScalarQueryParameter("specific_date", "DATE", specific_date),
            ]
        )

        bq_client = bigquery.Client()
        query_job = bq_client.query(query, job_config=job_config)

        return process_game_results(query_job, specific_date)


    except Exception as e:
        error_message = f"Unexpected error: {e}"
        logging.error(error_message)
        return [{"error": "Unexpected error", "message": error_message}]
    




def fetch_team_player_stats(team_name: str, limit: int = 100, specific_date: str = None) -> list:
    """
    Fetches player statistics for the most recent games of a team, optionally filtered by a specific date,
    ordered by play end time.

    Args:
        team_name: Team name (e.g., 'rangers', 'yankees').
        limit: Maximum number of records to return.
        specific_date: A specific date in 'YYYY-MM-DD' format.

    Returns:
        A list of dictionaries, each containing player stats, or an error object.
    """
    from google.cloud import bigquery
    from google.cloud.bigquery import exceptions
    import logging

    logging.basicConfig(level=logging.INFO)

    TEAMS = {
        'rangers': 140,
        'angels': 108,
        'astros': 117,
        'rays': 139,
        'blue_jays': 141,
        'yankees': 147,
        'orioles': 110,
        'red_sox': 111,
        'twins': 142,
        'white_sox': 145,
        'guardians': 114,
        'tigers': 116,
        'royals': 118,
        'padres': 135,
        'giants': 137,
        'diamondbacks': 109,
        'rockies': 115,
        'phillies': 143,
        'braves': 144,
        'marlins': 146,
        'nationals': 120,
        'mets': 121,
        'pirates': 134,
        'cardinals': 138,
        'brewers': 158,
        'cubs': 112,
        'reds': 113,
        'athletics': 133,
        'mariners': 136,
        'dodgers': 119,
    }

    FULL_TEAM_NAMES = {
        'texas rangers': 'rangers',
        'los angeles angels': 'angels',
        'houston astros': 'astros',
        'tampa bay rays': 'rays',
        'toronto blue jays': 'blue_jays',
        'new york yankees': 'yankees',
        'baltimore orioles': 'orioles',
        'boston red sox': 'red_sox',
        'minnesota twins': 'twins',
        'chicago white sox': 'white_sox',
        'cleveland guardians': 'guardians',
        'detroit tigers': 'tigers',
        'kansas city royals': 'royals',
        'san diego padres': 'padres',
        'san francisco giants': 'giants',
        'arizona diamondbacks': 'diamondbacks',
        'colorado rockies': 'rockies',
        'philadelphia phillies': 'phillies',
        'atlanta braves': 'braves',
        'miami marlins': 'marlins',
        'washington nationals': 'nationals',
        'new york mets': 'mets',
        'pittsburgh pirates': 'pirates',
        'st louis cardinals': 'cardinals',
        'milwaukee brewers': 'brewers',
        'chicago cubs': 'cubs',
        'cincinnati reds': 'reds',
        'oakland athletics': 'athletics',
        'seattle mariners': 'mariners',
        'los angeles dodgers': 'dodgers',
    }

    def _get_table_name(team_name: str) -> str:
        """Helper function to construct the table name."""
        cleaned_name = team_name.lower().strip()
        if cleaned_name in FULL_TEAM_NAMES:
            team_key = FULL_TEAM_NAMES[cleaned_name]
            return f"`gem-rush-007.{team_key}_mlb_data_2024`"
        for team_key in TEAMS:
            if team_key in cleaned_name:
                return f"`gem-rush-007.{team_key}_mlb_data_2024`"
        return f"`gem-rush-007.unknown_team_mlb_data_2024`"


    team_id = TEAMS[team_name]
    table_name = _get_table_name(team_name)

    try:
        query = f"""
            SELECT
                ps.player_id,
                r.full_name,
                g.official_date as game_date,
                ps.at_bats,
                ps.hits,
                ps.home_runs,
                ps.rbi,
                ps.walks,
                ps.strikeouts,
                ps.batting_average,
                ps.on_base_percentage,
                ps.slugging_percentage
            FROM
                {table_name}.player_stats AS ps
            JOIN
                {table_name}.roster AS r
                ON ps.player_id = r.player_id
            INNER JOIN
                {table_name}.games AS g
                ON ps.game_id = g.game_id
            INNER JOIN (
                SELECT
                    game_id,
                    MAX(end_time) as max_end_time
                FROM
                    {table_name}.plays
        """
        if specific_date:
            query += f" WHERE DATE(end_time) = @specific_date"
        query += f"""
                GROUP BY game_id
            ) AS subquery
            ON g.game_id = subquery.game_id
            WHERE
                (g.home_team_id = {team_id} OR g.away_team_id = {team_id})
        """
        if specific_date:
            query += f" AND g.official_date = @specific_date"
        query += f"""
           ORDER BY subquery.max_end_time DESC
           LIMIT @limit
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("limit", "INT64", limit),
                bigquery.ScalarQueryParameter("specific_date", "DATE", specific_date)
            ]
        )
        bq_client = bigquery.Client()
        query_job = bq_client.query(query, job_config=job_config)
        results = list(query_job.result())
        return [dict(row) for row in results]

    except Exception as e:
        error_message = f"Error fetching player stats for {team_name}: {e}"
        logging.error(error_message)
        return [{"error": "Data error", "message": error_message}]
    


def fetch_team_player_stats_by_opponent(team_name: str, opponent_team: str, limit: int = 100, specific_date: str = None) -> list:
    """
    Fetches player statistics for any team's games against a specific opponent,
    optionally filtered by a specific date.

    Args:
        team_name: Team name (e.g., 'rangers', 'yankees').
        opponent_team: Opponent team name.
        limit: Maximum records to return.
        specific_date: A specific date in 'YYYY-MM-DD' format.

    Returns:
        A list of dictionaries, each containing player stats, or an error object.
    """
    from google.cloud import bigquery
    from google.cloud.bigquery import exceptions
    import logging

    logging.basicConfig(level=logging.INFO)

    TEAMS = {
        'rangers': 140,
        'angels': 108,
        'astros': 117,
        'rays': 139,
        'blue_jays': 141,
        'yankees': 147,
        'orioles': 110,
        'red_sox': 111,
        'twins': 142,
        'white_sox': 145,
        'guardians': 114,
        'tigers': 116,
        'royals': 118,
        'padres': 135,
        'giants': 137,
        'diamondbacks': 109,
        'rockies': 115,
        'phillies': 143,
        'braves': 144,
        'marlins': 146,
        'nationals': 120,
        'mets': 121,
        'pirates': 134,
        'cardinals': 138,
        'brewers': 158,
        'cubs': 112,
        'reds': 113,
        'athletics': 133,
        'mariners': 136,
        'dodgers': 119,
    }

    FULL_TEAM_NAMES = {
        'texas rangers': 'rangers',
        'los angeles angels': 'angels',
        'houston astros': 'astros',
        'tampa bay rays': 'rays',
        'toronto blue jays': 'blue_jays',
        'new york yankees': 'yankees',
        'baltimore orioles': 'orioles',
        'boston red sox': 'red_sox',
        'minnesota twins': 'twins',
        'chicago white sox': 'white_sox',
        'cleveland guardians': 'guardians',
        'detroit tigers': 'tigers',
        'kansas city royals': 'royals',
        'san diego padres': 'padres',
        'san francisco giants': 'giants',
        'arizona diamondbacks': 'diamondbacks',
        'colorado rockies': 'rockies',
        'philadelphia phillies': 'phillies',
        'atlanta braves': 'braves',
        'miami marlins': 'marlins',
        'washington nationals': 'nationals',
        'new york mets': 'mets',
        'pittsburgh pirates': 'pirates',
        'st louis cardinals': 'cardinals',
        'milwaukee brewers': 'brewers',
        'chicago cubs': 'cubs',
        'cincinnati reds': 'reds',
        'oakland athletics': 'athletics',
        'seattle mariners': 'mariners',
        'los angeles dodgers': 'dodgers',
    }

    def _get_table_name(team_name: str) -> str:
        """Helper function to construct the table name."""
        cleaned_name = team_name.lower().strip()
        if cleaned_name in FULL_TEAM_NAMES:
            team_key = FULL_TEAM_NAMES[cleaned_name]
            return f"`gem-rush-007.{team_key}_mlb_data_2024`"
        for team_key in TEAMS:
            if team_key in cleaned_name:
                return f"`gem-rush-007.{team_key}_mlb_data_2024`"
        return f"`gem-rush-007.unknown_team_mlb_data_2024`"


    team_id = TEAMS[team_name]
    table_name = _get_table_name(team_name)

    try:
        query = f"""
        SELECT
            ps.player_id,
            r.full_name,
            g.official_date as game_date,
            ps.at_bats,
            ps.hits,
            ps.home_runs,
            ps.rbi,
            ps.walks,
            ps.strikeouts,
            ps.batting_average,
            ps.on_base_percentage,
            ps.slugging_percentage
        FROM
            {table_name}.player_stats AS ps
        JOIN
            {table_name}.roster AS r
            ON ps.player_id = r.player_id
        INNER JOIN
            {table_name}.games AS g
            ON ps.game_id = g.game_id
        INNER JOIN (
            SELECT
                game_id,
                MAX(end_time) as max_end_time
            FROM
                {table_name}.plays
        """
        if specific_date:
            query += f" WHERE DATE(end_time) = @specific_date"
        query += f"""
            GROUP BY game_id
        ) AS subquery
        ON g.game_id = subquery.game_id
        WHERE
            (g.home_team_id = {team_id} OR g.away_team_id = {team_id})
            AND ((g.home_team_name = @opponent_team) OR (g.away_team_name = @opponent_team))
        """
        if specific_date:
            query += f" AND g.official_date = @specific_date"
        query += f"""
        ORDER BY subquery.max_end_time DESC
        LIMIT @limit
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("opponent_team", "STRING", opponent_team),
                bigquery.ScalarQueryParameter("limit", "INT64", limit),
                bigquery.ScalarQueryParameter("specific_date", "DATE", specific_date)
            ]
        )
        bq_client = bigquery.Client()
        query_job = bq_client.query(query, job_config=job_config)
        results = list(query_job.result())
        return [dict(row) for row in results]

    except Exception as e:
        error_message = f"Error fetching player stats by opponent for {team_name}: {e}"
        logging.error(error_message)
        return [{"error": "Data error", "message": error_message}]
    


def fetch_team_player_stats_by_game_type(team_name: str, game_type: str, limit: int = 100, specific_date: str = None) -> list:
    """
    Fetches player statistics for any team by game type, optionally filtered by a specific date.

    Args:
        team_name: Team name (e.g., 'rangers', 'yankees').
        game_type: Game type (R, P, etc.).
        limit: Max records to return.
        specific_date: A specific date in 'YYYY-MM-DD' format.

    Returns:
        A list of dictionaries, each containing player stats, or an error object.
    """
    from google.cloud import bigquery
    from google.cloud.bigquery import exceptions
    import logging

    logging.basicConfig(level=logging.INFO)

    TEAMS = {
        'rangers': 140,
        'angels': 108,
        'astros': 117,
        'rays': 139,
        'blue_jays': 141,
        'yankees': 147,
        'orioles': 110,
        'red_sox': 111,
        'twins': 142,
        'white_sox': 145,
        'guardians': 114,
        'tigers': 116,
        'royals': 118,
        'padres': 135,
        'giants': 137,
        'diamondbacks': 109,
        'rockies': 115,
        'phillies': 143,
        'braves': 144,
        'marlins': 146,
        'nationals': 120,
        'mets': 121,
        'pirates': 134,
        'cardinals': 138,
        'brewers': 158,
        'cubs': 112,
        'reds': 113,
        'athletics': 133,
        'mariners': 136,
        'dodgers': 119,
    }

    FULL_TEAM_NAMES = {
        'texas rangers': 'rangers',
        'los angeles angels': 'angels',
        'houston astros': 'astros',
        'tampa bay rays': 'rays',
        'toronto blue jays': 'blue_jays',
        'new york yankees': 'yankees',
        'baltimore orioles': 'orioles',
        'boston red sox': 'red_sox',
        'minnesota twins': 'twins',
        'chicago white sox': 'white_sox',
        'cleveland guardians': 'guardians',
        'detroit tigers': 'tigers',
        'kansas city royals': 'royals',
        'san diego padres': 'padres',
        'san francisco giants': 'giants',
        'arizona diamondbacks': 'diamondbacks',
        'colorado rockies': 'rockies',
        'philadelphia phillies': 'phillies',
        'atlanta braves': 'braves',
        'miami marlins': 'marlins',
        'washington nationals': 'nationals',
        'new york mets': 'mets',
        'pittsburgh pirates': 'pirates',
        'st louis cardinals': 'cardinals',
        'milwaukee brewers': 'brewers',
        'chicago cubs': 'cubs',
        'cincinnati reds': 'reds',
        'oakland athletics': 'athletics',
        'seattle mariners': 'mariners',
        'los angeles dodgers': 'dodgers',
    }

    def _get_table_name(team_name: str) -> str:
        """Helper function to construct the table name."""
        cleaned_name = team_name.lower().strip()
        if cleaned_name in FULL_TEAM_NAMES:
            team_key = FULL_TEAM_NAMES[cleaned_name]
            return f"`gem-rush-007.{team_key}_mlb_data_2024`"
        for team_key in TEAMS:
            if team_key in cleaned_name:
                return f"`gem-rush-007.{team_key}_mlb_data_2024`"
        return f"`gem-rush-007.unknown_team_mlb_data_2024`"

    team_id = TEAMS[team_name]
    table_name = _get_table_name(team_name)

    try:
        query = f"""
        SELECT
            ps.player_id,
            g.official_date as game_date,
            r.full_name,
            ps.at_bats,
            ps.hits,
            ps.home_runs,
            ps.rbi,
            ps.walks,
            ps.strikeouts,
            ps.batting_average,
            ps.on_base_percentage,
            ps.slugging_percentage
        FROM
            {table_name}.player_stats AS ps
        JOIN
            {table_name}.roster AS r
            ON ps.player_id = r.player_id
        INNER JOIN
            {table_name}.games AS g
            ON ps.game_id = g.game_id
        INNER JOIN (
            SELECT
                game_id,
                MAX(end_time) as max_end_time
            FROM
                {table_name}.plays
        """
        if specific_date:
            query += f" WHERE DATE(end_time) = @specific_date"
        query += f"""
            GROUP BY game_id
        ) AS subquery
        ON g.game_id = subquery.game_id
        WHERE
            (g.home_team_id = {team_id} OR g.away_team_id = {team_id})
            AND g.game_type = @game_type
        """
        if specific_date:
            query += f" AND g.official_date = @specific_date"
        query += f"""
        ORDER BY subquery.max_end_time DESC
        LIMIT @limit
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("game_type", "STRING", game_type),
                bigquery.ScalarQueryParameter("limit", "INT64", limit),
                bigquery.ScalarQueryParameter("specific_date", "DATE", specific_date)
            ]
        )
        bq_client = bigquery.Client()
        query_job = bq_client.query(query, job_config=job_config)
        results = list(query_job.result())
        return [dict(row) for row in results]

    except Exception as e:
        error_message = f"Error fetching player stats by game type for {team_name}: {e}"
        logging.error(error_message)
        return [{"error": "Data error", "message": error_message}]
    


def fetch_team_plays(team_name: str, limit: int = 100, specific_date: str = None) -> list:
    """
    Fetches plays from any team's games, optionally filtered by a specific date.

    Args:
        team_name: Team name (e.g., 'rangers', 'yankees').
        limit: Max plays to return.
        specific_date: A specific date in 'YYYY-MM-DD' format.

    Returns:
        A list of dictionaries, each containing play data, or an error object.
    """
    from google.cloud import bigquery
    from google.cloud.bigquery import exceptions
    import logging

    logging.basicConfig(level=logging.INFO)

    TEAMS = {
        'rangers': 140,
        'angels': 108,
        'astros': 117,
        'rays': 139,
        'blue_jays': 141,
        'yankees': 147,
        'orioles': 110,
        'red_sox': 111,
        'twins': 142,
        'white_sox': 145,
        'guardians': 114,
        'tigers': 116,
        'royals': 118,
        'padres': 135,
        'giants': 137,
        'diamondbacks': 109,
        'rockies': 115,
        'phillies': 143,
        'braves': 144,
        'marlins': 146,
        'nationals': 120,
        'mets': 121,
        'pirates': 134,
        'cardinals': 138,
        'brewers': 158,
        'cubs': 112,
        'reds': 113,
        'athletics': 133,
        'mariners': 136,
        'dodgers': 119,
    }

    FULL_TEAM_NAMES = {
        'texas rangers': 'rangers',
        'los angeles angels': 'angels',
        'houston astros': 'astros',
        'tampa bay rays': 'rays',
        'toronto blue jays': 'blue_jays',
        'new york yankees': 'yankees',
        'baltimore orioles': 'orioles',
        'boston red sox': 'red_sox',
        'minnesota twins': 'twins',
        'chicago white sox': 'white_sox',
        'cleveland guardians': 'guardians',
        'detroit tigers': 'tigers',
        'kansas city royals': 'royals',
        'san diego padres': 'padres',
        'san francisco giants': 'giants',
        'arizona diamondbacks': 'diamondbacks',
        'colorado rockies': 'rockies',
        'philadelphia phillies': 'phillies',
        'atlanta braves': 'braves',
        'miami marlins': 'marlins',
        'washington nationals': 'nationals',
        'new york mets': 'mets',
        'pittsburgh pirates': 'pirates',
        'st louis cardinals': 'cardinals',
        'milwaukee brewers': 'brewers',
        'chicago cubs': 'cubs',
        'cincinnati reds': 'reds',
        'oakland athletics': 'athletics',
        'seattle mariners': 'mariners',
        'los angeles dodgers': 'dodgers',
    }

    def _get_table_name(team_name: str) -> str:
        """Helper function to construct the table name."""
        cleaned_name = team_name.lower().strip()
        if cleaned_name in FULL_TEAM_NAMES:
            team_key = FULL_TEAM_NAMES[cleaned_name]
            return f"`gem-rush-007.{team_key}_mlb_data_2024`"
        for team_key in TEAMS:
            if team_key in cleaned_name:
                return f"`gem-rush-007.{team_key}_mlb_data_2024`"
        return f"`gem-rush-007.unknown_team_mlb_data_2024`"

    team_id = TEAMS[team_name]
    table_name = _get_table_name(team_name)

    try:
        query = f"""
        SELECT
            p.play_id,
            p.inning,
            p.half_inning,
            p.event,
            p.event_type,
            p.description,
            p.rbi,
            p.is_scoring_play,
            r_batter.full_name as batter_name,
            r_pitcher.full_name as pitcher_name,
            p.start_time,
            p.end_time
        FROM
            {table_name}.plays AS p
        LEFT JOIN
            {table_name}.roster as r_batter
            ON p.batter_id = r_batter.player_id
        LEFT JOIN
            {table_name}.roster as r_pitcher
            ON p.pitcher_id = r_pitcher.player_id
        INNER JOIN
            {table_name}.games AS g
            ON p.game_id = g.game_id
        WHERE
            g.home_team_id = {team_id} OR g.away_team_id = {team_id}
        """
        if specific_date:
            query += f" AND g.official_date = @specific_date AND DATE(p.start_time) = @specific_date"
        query += f"""
        ORDER BY
            p.end_time DESC
        LIMIT @limit
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("limit", "INT64", limit),
                bigquery.ScalarQueryParameter("specific_date", "DATE", specific_date)
            ]
        )
        bq_client = bigquery.Client()
        query_job = bq_client.query(query, job_config=job_config)
        results = list(query_job.result())
        return [dict(row) for row in results]

    except Exception as e:
        error_message = f"Error fetching team plays for {team_name}: {e}"
        logging.error(error_message)
        return [{"error": "Data error", "message": error_message}]
    

def fetch_team_plays_by_opponent(team_name: str, opponent_team: str, limit: int = 100, specific_date: str = None) -> list:
    """
    Fetches plays from any team's games against a specific opponent, optionally filtered by a specific date.

    Args:
        team_name: Team name (e.g., 'rangers', 'yankees').
        opponent_team: Opponent team name.
        limit: Max plays to return.
        specific_date: A specific date in 'YYYY-MM-DD' format.

    Returns:
        A list of dictionaries, each containing play data, or an error object.
    """
    from google.cloud import bigquery
    from google.cloud.bigquery import exceptions
    import logging

    logging.basicConfig(level=logging.INFO)

    TEAMS = {
        'rangers': 140,
        'angels': 108,
        'astros': 117,
        'rays': 139,
        'blue_jays': 141,
        'yankees': 147,
        'orioles': 110,
        'red_sox': 111,
        'twins': 142,
        'white_sox': 145,
        'guardians': 114,
        'tigers': 116,
        'royals': 118,
        'padres': 135,
        'giants': 137,
        'diamondbacks': 109,
        'rockies': 115,
        'phillies': 143,
        'braves': 144,
        'marlins': 146,
        'nationals': 120,
        'mets': 121,
        'pirates': 134,
        'cardinals': 138,
        'brewers': 158,
        'cubs': 112,
        'reds': 113,
        'athletics': 133,
        'mariners': 136,
        'dodgers': 119,
    }

    FULL_TEAM_NAMES = {
        'texas rangers': 'rangers',
        'los angeles angels': 'angels',
        'houston astros': 'astros',
        'tampa bay rays': 'rays',
        'toronto blue jays': 'blue_jays',
        'new york yankees': 'yankees',
        'baltimore orioles': 'orioles',
        'boston red sox': 'red_sox',
        'minnesota twins': 'twins',
        'chicago white sox': 'white_sox',
        'cleveland guardians': 'guardians',
        'detroit tigers': 'tigers',
        'kansas city royals': 'royals',
        'san diego padres': 'padres',
        'san francisco giants': 'giants',
        'arizona diamondbacks': 'diamondbacks',
        'colorado rockies': 'rockies',
        'philadelphia phillies': 'phillies',
        'atlanta braves': 'braves',
        'miami marlins': 'marlins',
        'washington nationals': 'nationals',
        'new york mets': 'mets',
        'pittsburgh pirates': 'pirates',
        'st louis cardinals': 'cardinals',
        'milwaukee brewers': 'brewers',
        'chicago cubs': 'cubs',
        'cincinnati reds': 'reds',
        'oakland athletics': 'athletics',
        'seattle mariners': 'mariners',
        'los angeles dodgers': 'dodgers',
    }

    def _get_table_name(team_name: str) -> str:
        """Helper function to construct the table name."""
        cleaned_name = team_name.lower().strip()
        if cleaned_name in FULL_TEAM_NAMES:
            team_key = FULL_TEAM_NAMES[cleaned_name]
            return f"`gem-rush-007.{team_key}_mlb_data_2024`"
        for team_key in TEAMS:
            if team_key in cleaned_name:
                return f"`gem-rush-007.{team_key}_mlb_data_2024`"
        return f"`gem-rush-007.unknown_team_mlb_data_2024`"

    team_id = TEAMS[team_name]
    table_name = _get_table_name(team_name)

    try:
        query = f"""
        SELECT
            p.play_id,
            p.inning,
            p.half_inning,
            p.event,
            p.event_type,
            p.description,
            p.rbi,
            p.is_scoring_play,
            r_batter.full_name as batter_name,
            r_pitcher.full_name as pitcher_name,
            p.start_time,
            p.end_time
        FROM
            {table_name}.plays AS p
        LEFT JOIN
            {table_name}.roster as r_batter
            ON p.batter_id = r_batter.player_id
        LEFT JOIN
            {table_name}.roster as r_pitcher
            ON p.pitcher_id = r_pitcher.player_id
        INNER JOIN
            {table_name}.games AS g
            ON p.game_id = g.game_id
        WHERE
            (g.home_team_id = {team_id} OR g.away_team_id = {team_id})
            AND ((g.home_team_name = @opponent_team) OR (g.away_team_name = @opponent_team))
        """
        if specific_date:
            query += f" AND g.official_date = @specific_date AND DATE(p.start_time) = @specific_date"
        query += f"""
        ORDER BY
            p.end_time DESC, p.start_time ASC
        LIMIT @limit
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("opponent_team", "STRING", opponent_team),
                bigquery.ScalarQueryParameter("limit", "INT64", limit),
                bigquery.ScalarQueryParameter("specific_date", "DATE", specific_date)
            ]
        )
        bq_client = bigquery.Client()
        query_job = bq_client.query(query, job_config=job_config)
        results = list(query_job.result())
        return [dict(row) for row in results]

    except Exception as e:
        error_message = f"Error fetching team plays by opponent for {team_name}: {e}"
        logging.error(error_message)
        return [{"error": "Data error", "message": error_message}]


def fetch_team_plays_by_game_type(team_name: str, game_type: str, limit: int = 100, specific_date: str = None) -> list:
    """
    Fetches plays from any team's games by game type, optionally filtered by a specific date.

    Args:
        team_name: Team name.
        game_type: Game type (R, P, etc.).
        limit: Max plays to return.
        specific_date: A specific date in 'YYYY-MM-DD' format.

    Returns:
        A list of dictionaries, each containing play details, or an error object.
    """
    from google.cloud import bigquery
    from google.cloud.bigquery import exceptions
    import logging

    logging.basicConfig(level=logging.INFO)

    TEAMS = {
        'rangers': 140,
        'angels': 108,
        'astros': 117,
        'rays': 139,
        'blue_jays': 141,
        'yankees': 147,
        'orioles': 110,
        'red_sox': 111,
        'twins': 142,
        'white_sox': 145,
        'guardians': 114,
        'tigers': 116,
        'royals': 118,
        'padres': 135,
        'giants': 137,
        'diamondbacks': 109,
        'rockies': 115,
        'phillies': 143,
        'braves': 144,
        'marlins': 146,
        'nationals': 120,
        'mets': 121,
        'pirates': 134,
        'cardinals': 138,
        'brewers': 158,
        'cubs': 112,
        'reds': 113,
        'athletics': 133,
        'mariners': 136,
        'dodgers': 119,
    }
    FULL_TEAM_NAMES = {
        'texas rangers': 'rangers',
        'los angeles angels': 'angels',
        'houston astros': 'astros',
        'tampa bay rays': 'rays',
        'toronto blue jays': 'blue_jays',
        'new york yankees': 'yankees',
        'baltimore orioles': 'orioles',
        'boston red sox': 'red_sox',
        'minnesota twins': 'twins',
        'chicago white sox': 'white_sox',
        'cleveland guardians': 'guardians',
        'detroit tigers': 'tigers',
        'kansas city royals': 'royals',
        'san diego padres': 'padres',
        'san francisco giants': 'giants',
        'arizona diamondbacks': 'diamondbacks',
        'colorado rockies': 'rockies',
        'philadelphia phillies': 'phillies',
        'atlanta braves': 'braves',
        'miami marlins': 'marlins',
        'washington nationals': 'nationals',
        'new york mets': 'mets',
        'pittsburgh pirates': 'pirates',
        'st louis cardinals': 'cardinals',
        'milwaukee brewers': 'brewers',
        'chicago cubs': 'cubs',
        'cincinnati reds': 'reds',
        'oakland athletics': 'athletics',
        'seattle mariners': 'mariners',
        'los angeles dodgers': 'dodgers',
    }

    def _get_table_name(team_name: str) -> str:
        """Helper function to construct the table name."""
        cleaned_name = team_name.lower().strip()
        if cleaned_name in FULL_TEAM_NAMES:
            team_key = FULL_TEAM_NAMES[cleaned_name]
            return f"`gem-rush-007.{team_key}_mlb_data_2024`"
        for team_key in TEAMS:
            if team_key in cleaned_name:
                return f"`gem-rush-007.{team_key}_mlb_data_2024`"
        return f"`gem-rush-007.unknown_team_mlb_data_2024`"

    team_id = TEAMS[team_name]
    table_name = _get_table_name(team_name)

    try:
        query = f"""
        SELECT
            p.play_id,
            p.inning,
            p.half_inning,
            p.event,
            p.event_type,
            p.description,
            p.rbi,
            p.is_scoring_play,
            r_batter.full_name as batter_name,
            r_pitcher.full_name as pitcher_name,
            p.start_time,
            p.end_time
        FROM
            {table_name}.plays AS p
        LEFT JOIN
            {table_name}.roster as r_batter
            ON p.batter_id = r_batter.player_id
        LEFT JOIN
            {table_name}.roster as r_pitcher
            ON p.pitcher_id = r_pitcher.player_id
        INNER JOIN
            {table_name}.games AS g
            ON p.game_id = g.game_id
        WHERE
            (g.home_team_id = {team_id} OR g.away_team_id = {team_id})
            AND g.game_type = @game_type
        """
        if specific_date:
            query += f" AND g.official_date = @specific_date AND DATE(p.start_time) = @specific_date"

        query += f"""
        ORDER BY
            p.end_time DESC, p.start_time ASC
        LIMIT @limit
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("game_type", "STRING", game_type),
                bigquery.ScalarQueryParameter("limit", "INT64", limit),
                bigquery.ScalarQueryParameter("specific_date", "DATE", specific_date)
            ]
        )
        bq_client = bigquery.Client()
        query_job = bq_client.query(query, job_config=job_config)
        results = list(query_job.result())
        return [dict(row) for row in results]

    except Exception as e:
        error_message = f"Error fetching team plays by game type for {team_name}: {e}"
        logging.error(error_message)
        return [{"error": "Data error", "message": error_message}]



def fetch_team_games_by_opponent(team_name: str, opponent_team: str = 'New York Yankees', limit: int = 2, specific_date: str = None) -> list:
    """
    Fetches any team's games against specific opponent, optionally filtered by a specific date.

    Args:
        team_name: Team name.
        opponent_team: Opponent team name.
        limit: Max games to return.
        specific_date: A specific date in 'YYYY-MM-DD' format.

    Returns:
        A list of dictionaries, each containing game details, or an error object.
    """
    from google.cloud import bigquery
    from google.cloud.bigquery import exceptions
    import logging

    logging.basicConfig(level=logging.INFO)

    TEAMS = {
        'rangers': 140,
        'angels': 108,
        'astros': 117,
        'rays': 139,
        'blue_jays': 141,
        'yankees': 147,
        'orioles': 110,
        'red_sox': 111,
        'twins': 142,
        'white_sox': 145,
        'guardians': 114,
        'tigers': 116,
        'royals': 118,
        'padres': 135,
        'giants': 137,
        'diamondbacks': 109,
        'rockies': 115,
        'phillies': 143,
        'braves': 144,
        'marlins': 146,
        'nationals': 120,
        'mets': 121,
        'pirates': 134,
        'cardinals': 138,
        'brewers': 158,
        'cubs': 112,
        'reds': 113,
        'athletics': 133,
        'mariners': 136,
        'dodgers': 119,
    }

    FULL_TEAM_NAMES = {
        'texas rangers': 'rangers',
        'los angeles angels': 'angels',
        'houston astros': 'astros',
        'tampa bay rays': 'rays',
        'toronto blue jays': 'blue_jays',
        'new york yankees': 'yankees',
        'baltimore orioles': 'orioles',
        'boston red sox': 'red_sox',
        'minnesota twins': 'twins',
        'chicago white sox': 'white_sox',
        'cleveland guardians': 'guardians',
        'detroit tigers': 'tigers',
        'kansas city royals': 'royals',
        'san diego padres': 'padres',
        'san francisco giants': 'giants',
        'arizona diamondbacks': 'diamondbacks',
        'colorado rockies': 'rockies',
        'philadelphia phillies': 'phillies',
        'atlanta braves': 'braves',
        'miami marlins': 'marlins',
        'washington nationals': 'nationals',
        'new york mets': 'mets',
        'pittsburgh pirates': 'pirates',
        'st louis cardinals': 'cardinals',
        'milwaukee brewers': 'brewers',
        'chicago cubs': 'cubs',
        'cincinnati reds': 'reds',
        'oakland athletics': 'athletics',
        'seattle mariners': 'mariners',
        'los angeles dodgers': 'dodgers',
    }
    def _get_table_name(team_name: str) -> str:
        """Helper function to construct the table name."""
        cleaned_name = team_name.lower().strip()
        if cleaned_name in FULL_TEAM_NAMES:
            team_key = FULL_TEAM_NAMES[cleaned_name]
            return f"`gem-rush-007.{team_key}_mlb_data_2024`"
        for team_key in TEAMS:
            if team_key in cleaned_name:
                return f"`gem-rush-007.{team_key}_mlb_data_2024`"
        return f"`gem-rush-007.unknown_team_mlb_data_2024`"

    def process_game_results(query_job, specific_date=None):
        """Helper to process results, validate, and check against specific_date."""
        results = []
        for row in query_job:
            row_dict = dict(row)
            if row_dict.get('official_date') and not isinstance(row_dict['official_date'], str):
                row_dict['official_date'] = row_dict['official_date'].strftime('%Y-%m-%d')
            if specific_date and row_dict.get('official_date') != specific_date:
                continue
            required_fields = ['game_id', 'official_date']
            if not all(row_dict.get(field) for field in required_fields):
                continue
            numeric_fields = ['home_score', 'away_score', 'team_margin']
            for field in numeric_fields:
                try:
                    if row_dict.get(field) is not None:
                        row_dict[field] = int(row_dict[field])
                except (TypeError, ValueError):
                    row_dict[field] = None
            results.append(row_dict)
        return results

    team_id = TEAMS[team_name]
    table_name = _get_table_name(team_name)

    try:
        query = f"""
        SELECT
            g.game_id,
            g.official_date,
            g.home_team_id,
            g.home_team_name,
            g.away_team_id,
            g.away_team_name,
            g.home_score,
            g.away_score,
            g.venue_name,
            g.status,
            {team_name}_win as team_win,
            {team_name}_margin as team_margin
        FROM
            {table_name}.games AS g
        INNER JOIN
            (SELECT
                game_id,
                MAX(end_time) AS max_end_time
            FROM
                {table_name}.plays
        """
        if specific_date:
          query += f"WHERE DATE(start_time) = '{specific_date}'"

        query += f"""
            GROUP BY game_id
            ORDER BY max_end_time DESC
            LIMIT @limit
        ) AS subquery
        ON g.game_id = subquery.game_id
        WHERE
            (g.home_team_id = {team_id} OR g.away_team_id = {team_id})
            AND ((g.home_team_name = @opponent_team) OR (g.away_team_name = @opponent_team))
        """
        if specific_date:
            query += f" AND g.official_date = '{specific_date}'"

        query += f"""
        ORDER BY subquery.max_end_time DESC
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("limit", "INT64", limit),
                bigquery.ScalarQueryParameter("opponent_team", "STRING", opponent_team)
            ]
        )
        bq_client = bigquery.Client()
        query_job = bq_client.query(query, job_config=job_config)
        return process_game_results(query_job)

    except Exception as e:
        error_message = f"Error fetching team games by opponent for {team_name}: {e}"
        logging.error(error_message)
        return [{"error": "Data error", "message": error_message}]



def fetch_team_games_by_type(team_name: str, game_type: str = 'R', limit: int = 2, specific_date: str = None) -> list:
    """
    Fetches any team's games by game type.

    Args:
        team_name: Team name.
        game_type: Game type (R=Regular, L=League Championship, etc.).
        limit: Max games to return.
        specific_date: A specific date in 'YYYY-MM-DD' format.

    Returns:
        A list of dictionaries, each containing game details, or an error object.
    """
    from google.cloud import bigquery
    from google.cloud.bigquery import exceptions
    import logging

    logging.basicConfig(level=logging.INFO)

    TEAMS = {
        'rangers': 140,
        'angels': 108,
        'astros': 117,
        'rays': 139,
        'blue_jays': 141,
        'yankees': 147,
        'orioles': 110,
        'red_sox': 111,
        'twins': 142,
        'white_sox': 145,
        'guardians': 114,
        'tigers': 116,
        'royals': 118,
        'padres': 135,
        'giants': 137,
        'diamondbacks': 109,
        'rockies': 115,
        'phillies': 143,
        'braves': 144,
        'marlins': 146,
        'nationals': 120,
        'mets': 121,
        'pirates': 134,
        'cardinals': 138,
        'brewers': 158,
        'cubs': 112,
        'reds': 113,
        'athletics': 133,
        'mariners': 136,
        'dodgers': 119,
    }
    FULL_TEAM_NAMES = {
        'texas rangers': 'rangers',
        'los angeles angels': 'angels',
        'houston astros': 'astros',
        'tampa bay rays': 'rays',
        'toronto blue jays': 'blue_jays',
        'new york yankees': 'yankees',
        'baltimore orioles': 'orioles',
        'boston red sox': 'red_sox',
        'minnesota twins': 'twins',
        'chicago white sox': 'white_sox',
        'cleveland guardians': 'guardians',
        'detroit tigers': 'tigers',
        'kansas city royals': 'royals',
        'san diego padres': 'padres',
        'san francisco giants': 'giants',
        'arizona diamondbacks': 'diamondbacks',
        'colorado rockies': 'rockies',
        'philadelphia phillies': 'phillies',
        'atlanta braves': 'braves',
        'miami marlins': 'marlins',
        'washington nationals': 'nationals',
        'new york mets': 'mets',
        'pittsburgh pirates': 'pirates',
        'st louis cardinals': 'cardinals',
        'milwaukee brewers': 'brewers',
        'chicago cubs': 'cubs',
        'cincinnati reds': 'reds',
        'oakland athletics': 'athletics',
        'seattle mariners': 'mariners',
        'los angeles dodgers': 'dodgers',
    }

    def _get_table_name(team_name: str) -> str:
        """Helper function to construct the table name."""
        cleaned_name = team_name.lower().strip()
        if cleaned_name in FULL_TEAM_NAMES:
            team_key = FULL_TEAM_NAMES[cleaned_name]
            return f"`gem-rush-007.{team_key}_mlb_data_2024`"
        for team_key in TEAMS:
            if team_key in cleaned_name:
                return f"`gem-rush-007.{team_key}_mlb_data_2024`"
        return f"`gem-rush-007.unknown_team_mlb_data_2024`"

    def process_game_results(query_job, specific_date=None):
        """Helper to process results, validate, and check against specific_date."""
        results = []
        for row in query_job:
            row_dict = dict(row)
            if row_dict.get('official_date') and not isinstance(row_dict['official_date'], str):
                row_dict['official_date'] = row_dict['official_date'].strftime('%Y-%m-%d')
            if specific_date and row_dict.get('official_date') != specific_date:
                continue
            required_fields = ['game_id', 'official_date']
            if not all(row_dict.get(field) for field in required_fields):
                continue
            numeric_fields = ['home_score', 'away_score', 'team_margin']
            for field in numeric_fields:
                try:
                    if row_dict.get(field) is not None:
                        row_dict[field] = int(row_dict[field])
                except (TypeError, ValueError):
                    row_dict[field] = None
            results.append(row_dict)
        return results

    team_id = TEAMS[team_name]
    table_name = _get_table_name(team_name)

    try:
        query = f"""
        SELECT
            g.game_id,
            g.official_date,
            g.home_team_id,
            g.home_team_name,
            g.away_team_id,
            g.away_team_name,
            g.home_score,
            g.away_score,
            g.venue_name,
            g.status,
            {team_name}_win as team_win,
            {team_name}_margin as team_margin
        FROM
              {table_name}.games AS g
        INNER JOIN
              (SELECT
                game_id,
                MAX(end_time) AS max_end_time
              FROM
                  {table_name}.plays
        """
        if specific_date:
            query += f" WHERE DATE(start_time) = @specific_date"
        query += f"""
              GROUP BY game_id
            ) AS subquery
              ON g.game_id = subquery.game_id
        WHERE
            (g.home_team_id = {team_id} OR g.away_team_id = {team_id})
            AND g.game_type = @game_type
        """
        if specific_date:
            query += f" AND g.official_date = @specific_date"
        query += f"""
        ORDER BY subquery.max_end_time DESC
        LIMIT @limit
        """

        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("limit", "INT64", limit),
                bigquery.ScalarQueryParameter("game_type", "STRING", game_type),
                bigquery.ScalarQueryParameter("specific_date", "DATE", specific_date)
            ]
        )

        bq_client = bigquery.Client()
        query_job = bq_client.query(query, job_config=job_config)
        return process_game_results(query_job)

    except Exception as e:
        error_message = f"Error fetching team games by type for {team_name}: {e}"
        logging.error(error_message)
        return [{"error": "Data error", "message": error_message}]



def fetch_player_game_stats(team_name: str, game_ids: list[str] = None, limit: int = 100, player_ids: list[str] = None, specific_date: str = None) -> list:
    """
    Fetches player statistics for any team's games/players, filtered by date, game IDs, or player IDs.

    Args:
        team_name: Team name.
        game_ids: Optional list of game IDs.
        limit: Maximum number of records.
        player_ids: Optional list of player IDs.
        specific_date: A specific date in 'YYYY-MM-DD' format.

    Returns:
        A list of dictionaries, each containing player stats, or an error object.
    """
    from google.cloud import bigquery
    from google.cloud.bigquery import exceptions
    import logging
    from datetime import datetime

    logging.basicConfig(level=logging.INFO)

    TEAMS = {
        'rangers': 140,
        'angels': 108,
        'astros': 117,
        'rays': 139,
        'blue_jays': 141,
        'yankees': 147,
        'orioles': 110,
        'red_sox': 111,
        'twins': 142,
        'white_sox': 145,
        'guardians': 114,
        'tigers': 116,
        'royals': 118,
        'padres': 135,
        'giants': 137,
        'diamondbacks': 109,
        'rockies': 115,
        'phillies': 143,
        'braves': 144,
        'marlins': 146,
        'nationals': 120,
        'mets': 121,
        'pirates': 134,
        'cardinals': 138,
        'brewers': 158,
        'cubs': 112,
        'reds': 113,
        'athletics': 133,
        'mariners': 136,
        'dodgers': 119,
    }

    FULL_TEAM_NAMES = {
        'texas rangers': 'rangers',
        'los angeles angels': 'angels',
        'houston astros': 'astros',
        'tampa bay rays': 'rays',
        'toronto blue jays': 'blue_jays',
        'new york yankees': 'yankees',
        'baltimore orioles': 'orioles',
        'boston red sox': 'red_sox',
        'minnesota twins': 'twins',
        'chicago white sox': 'white_sox',
        'cleveland guardians': 'guardians',
        'detroit tigers': 'tigers',
        'kansas city royals': 'royals',
        'san diego padres': 'padres',
        'san francisco giants': 'giants',
        'arizona diamondbacks': 'diamondbacks',
        'colorado rockies': 'rockies',
        'philadelphia phillies': 'phillies',
        'atlanta braves': 'braves',
        'miami marlins': 'marlins',
        'washington nationals': 'nationals',
        'new york mets': 'mets',
        'pittsburgh pirates': 'pirates',
        'st louis cardinals': 'cardinals',
        'milwaukee brewers': 'brewers',
        'chicago cubs': 'cubs',
        'cincinnati reds': 'reds',
        'oakland athletics': 'athletics',
        'seattle mariners': 'mariners',
        'los angeles dodgers': 'dodgers',
    }
    def _get_table_name(team_name: str) -> str:
        """Helper function to construct the table name."""
        cleaned_name = team_name.lower().strip()
        if cleaned_name in FULL_TEAM_NAMES:
            team_key = FULL_TEAM_NAMES[cleaned_name]
            return f"`gem-rush-007.{team_key}_mlb_data_2024`"
        for team_key in TEAMS:
            if team_key in cleaned_name:
                return f"`gem-rush-007.{team_key}_mlb_data_2024`"
        return f"`gem-rush-007.unknown_team_mlb_data_2024`"


    team_id = TEAMS[team_name]
    table_name = _get_table_name(team_name)

    if game_ids is not None and not isinstance(game_ids, list):
        raise ValueError("game_ids must be a list or None")
    if player_ids is not None and not isinstance(player_ids, list):
        raise ValueError("player_ids must be a list or None")

    try:
        query = f"""
        SELECT
            ps.player_id,
            g.official_date as game_date,
            r.full_name,
            ps.at_bats,
            ps.hits,
            ps.home_runs,
            ps.rbi,
            ps.walks,
            ps.strikeouts,
            ps.batting_average,
            ps.on_base_percentage,
            ps.slugging_percentage
        FROM
            {table_name}.player_stats AS ps
        JOIN
            {table_name}.roster AS r
        ON
            ps.player_id = r.player_id
        INNER JOIN
            {table_name}.games AS g
            ON ps.game_id = g.game_id
        INNER JOIN (
          SELECT
              game_id,
              MAX(end_time) as max_end_time
          FROM
             {table_name}.plays
        """
        if specific_date:
            query += f" WHERE DATE(end_time) = @specific_date"

        query += f"""
          GROUP BY game_id
         ) AS subquery
         ON g.game_id = subquery.game_id
        WHERE
            (g.home_team_id = {team_id} OR g.away_team_id = {team_id})
        """

        if specific_date:
            query += f" AND g.official_date = @specific_date"

        query_params = [
            bigquery.ScalarQueryParameter("limit", "INT64", limit),
            bigquery.ScalarQueryParameter("specific_date", "DATE", specific_date)
        ]
        where_conditions = []
        if game_ids:
            where_conditions.append("ps.game_id IN UNNEST(@game_id_list)")
            query_params.append(
                bigquery.ArrayQueryParameter("game_id_list", "STRING", game_ids)
            )
        if player_ids:
            where_conditions.append("ps.player_id IN UNNEST(@player_id_list)")
            query_params.append(
                bigquery.ArrayQueryParameter("player_id_list", "STRING", player_ids)
            )
        if where_conditions:
            query += "\n AND " + " AND ".join(where_conditions)

        query += f"""
        ORDER BY subquery.max_end_time DESC
        LIMIT @limit
        """
        job_config = bigquery.QueryJobConfig(query_parameters=query_params)
        bq_client = bigquery.Client()
        query_job = bq_client.query(query, job_config=job_config)

        results = []
        for row in query_job:
            row_dict = dict(row)

            if not all(row_dict.get(field) for field in ['player_id', 'full_name']):
                logging.warning(f"Skipping record with missing required information: {row_dict.get('player_id', 'Unknown Player')}")
                continue

            numeric_fields = [
                'at_bats', 'hits', 'home_runs', 'rbi', 'walks', 'strikeouts',
                'batting_average', 'on_base_percentage', 'slugging_percentage'
            ]

            for field in numeric_fields:
                try:
                    if row_dict.get(field) is not None:
                        if field in ['batting_average', 'on_base_percentage', 'slugging_percentage']:
                            row_dict[field] = float(row_dict[field])
                        else:
                            row_dict[field] = int(row_dict[field])
                except (TypeError, ValueError) as e:
                    logging.warning(f"Invalid {field} data for player {row_dict['full_name']}: {e}")
                    row_dict[field] = None
            results.append(row_dict)
        return results

    except Exception as e:
        error_message = f"Error fetching player game stats for {team_name}: {e}"
        logging.error(error_message)
        return [{"error": "Data error", "message": error_message}]


def fetch_player_plays(player_name: str, team_name: str, limit: int = 100, specific_date: str = None) -> list:
    """
    Fetches play-by-play data for a specific player, optionally filtered by date.

    Args:
        player_name: Full name of the player.
        team_name: Team name.
        limit: Maximum number of plays.
        specific_date: A specific date in 'YYYY-MM-DD' format.

    Returns:
        A list of dictionaries, each containing play details, or an error object.
    """
    from google.cloud import bigquery
    from google.cloud.bigquery import exceptions
    import logging
    from datetime import datetime

    logging.basicConfig(level=logging.INFO)

    TEAMS = {
        'rangers': 140,
        'angels': 108,
        'astros': 117,
        'rays': 139,
        'blue_jays': 141,
        'yankees': 147,
        'orioles': 110,
        'red_sox': 111,
        'twins': 142,
        'white_sox': 145,
        'guardians': 114,
        'tigers': 116,
        'royals': 118,
        'padres': 135,
        'giants': 137,
        'diamondbacks': 109,
        'rockies': 115,
        'phillies': 143,
        'braves': 144,
        'marlins': 146,
        'nationals': 120,
        'mets': 121,
        'pirates': 134,
        'cardinals': 138,
        'brewers': 158,
        'cubs': 112,
        'reds': 113,
        'athletics': 133,
        'mariners': 136,
        'dodgers': 119,
    }
    FULL_TEAM_NAMES = {
        'texas rangers': 'rangers',
        'los angeles angels': 'angels',
        'houston astros': 'astros',
        'tampa bay rays': 'rays',
        'toronto blue jays': 'blue_jays',
        'new york yankees': 'yankees',
        'baltimore orioles': 'orioles',
        'boston red sox': 'red_sox',
        'minnesota twins': 'twins',
        'chicago white sox': 'white_sox',
        'cleveland guardians': 'guardians',
        'detroit tigers': 'tigers',
        'kansas city royals': 'royals',
        'san diego padres': 'padres',
        'san francisco giants': 'giants',
        'arizona diamondbacks': 'diamondbacks',
        'colorado rockies': 'rockies',
        'philadelphia phillies': 'phillies',
        'atlanta braves': 'braves',
        'miami marlins': 'marlins',
        'washington nationals': 'nationals',
        'new york mets': 'mets',
        'pittsburgh pirates': 'pirates',
        'st louis cardinals': 'cardinals',
        'milwaukee brewers': 'brewers',
        'chicago cubs': 'cubs',
        'cincinnati reds': 'reds',
        'oakland athletics': 'athletics',
        'seattle mariners': 'mariners',
        'los angeles dodgers': 'dodgers',
    }

    def _get_table_name(team_name: str) -> str:
        """Helper function to construct the table name."""
        cleaned_name = team_name.lower().strip()
        if cleaned_name in FULL_TEAM_NAMES:
            team_key = FULL_TEAM_NAMES[cleaned_name]
            return f"`gem-rush-007.{team_key}_mlb_data_2024`"
        for team_key in TEAMS:
            if team_key in cleaned_name:
                return f"`gem-rush-007.{team_key}_mlb_data_2024`"
        return f"`gem-rush-007.unknown_team_mlb_data_2024`"

    team_id = TEAMS[team_name]
    table_name = _get_table_name(team_name)

    try:
        query = f"""
        SELECT
            p.play_id,
            p.inning,
            p.half_inning,
            p.event,
            p.event_type,
            p.description,
            p.start_time,
            g.official_date as game_date
        FROM
            {table_name}.plays AS p
        INNER JOIN
            {table_name}.games AS g
            ON p.game_id = g.game_id
        INNER JOIN
            {table_name}.roster AS r
            ON (p.batter_id = r.player_id OR p.pitcher_id = r.player_id)
        WHERE
            r.full_name = @player_name
            AND (g.home_team_id = {team_id} OR g.away_team_id = {team_id})
        """
        if specific_date:
            query += f" AND g.official_date = @specific_date AND DATE(p.start_time) = @specific_date"

        query += f"""
        ORDER BY
            p.end_time DESC,
            p.start_time ASC
        LIMIT @limit
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("player_name", "STRING", player_name),
                bigquery.ScalarQueryParameter("limit", "INT64", limit),
                bigquery.ScalarQueryParameter("specific_date", "DATE", specific_date)
            ]
        )
        bq_client = bigquery.Client()
        query_job = bq_client.query(query, job_config=job_config)
        results = list(query_job.result())

        formatted_results = []
        for row in results:
            row_dict = dict(row)
            if 'start_time' in row_dict and row_dict['start_time'] and isinstance(row_dict['start_time'], datetime):
                row_dict['start_time'] = row_dict['start_time'].isoformat()
            if 'game_date' in row_dict and row_dict['game_date'] and isinstance(row_dict['game_date'], datetime.date):
                row_dict['game_date'] = row_dict['game_date'].isoformat()
            formatted_results.append(row_dict)
        return formatted_results

    except Exception as e:
        error_message = f"Error fetching player plays for {player_name} on {team_name}: {e}"
        logging.error(error_message)
        return [{"error": "Data error", "message": error_message}]



def fetch_player_plays_by_opponent(player_name: str, team_name: str, opponent_team: str, limit: int = 100, specific_date: str = None) -> list:
    """
    Fetches play-by-play data for a player against a specific opponent, optionally filtered by date.

    Args:
        player_name: Player's full name.
        team_name: Team name.
        opponent_team: Opponent team name.
        limit: Max number of plays.
        specific_date: A specific date in 'YYYY-MM-DD' format.

    Returns:
        A list of dictionaries with play details, or an error object.
    """
    from google.cloud import bigquery
    from google.cloud.bigquery import exceptions
    import logging
    from datetime import datetime

    logging.basicConfig(level=logging.INFO)

    TEAMS = {
        'rangers': 140,
        'angels': 108,
        'astros': 117,
        'rays': 139,
        'blue_jays': 141,
        'yankees': 147,
        'orioles': 110,
        'red_sox': 111,
        'twins': 142,
        'white_sox': 145,
        'guardians': 114,
        'tigers': 116,
        'royals': 118,
        'padres': 135,
        'giants': 137,
        'diamondbacks': 109,
        'rockies': 115,
        'phillies': 143,
        'braves': 144,
        'marlins': 146,
        'nationals': 120,
        'mets': 121,
        'pirates': 134,
        'cardinals': 138,
        'brewers': 158,
        'cubs': 112,
        'reds': 113,
        'athletics': 133,
        'mariners': 136,
        'dodgers': 119,
    }
    FULL_TEAM_NAMES = {
        'texas rangers': 'rangers',
        'los angeles angels': 'angels',
        'houston astros': 'astros',
        'tampa bay rays': 'rays',
        'toronto blue jays': 'blue_jays',
        'new york yankees': 'yankees',
        'baltimore orioles': 'orioles',
        'boston red sox': 'red_sox',
        'minnesota twins': 'twins',
        'chicago white sox': 'white_sox',
        'cleveland guardians': 'guardians',
        'detroit tigers': 'tigers',
        'kansas city royals': 'royals',
        'san diego padres': 'padres',
        'san francisco giants': 'giants',
        'arizona diamondbacks': 'diamondbacks',
        'colorado rockies': 'rockies',
        'philadelphia phillies': 'phillies',
        'atlanta braves': 'braves',
        'miami marlins': 'marlins',
        'washington nationals': 'nationals',
        'new york mets': 'mets',
        'pittsburgh pirates': 'pirates',
        'st louis cardinals': 'cardinals',
        'milwaukee brewers': 'brewers',
        'chicago cubs': 'cubs',
        'cincinnati reds': 'reds',
        'oakland athletics': 'athletics',
        'seattle mariners': 'mariners',
        'los angeles dodgers': 'dodgers',
    }
    def _get_table_name(team_name: str) -> str:
        """Helper function to construct the table name."""
        cleaned_name = team_name.lower().strip()
        if cleaned_name in FULL_TEAM_NAMES:
            team_key = FULL_TEAM_NAMES[cleaned_name]
            return f"`gem-rush-007.{team_key}_mlb_data_2024`"
        for team_key in TEAMS:
            if team_key in cleaned_name:
                return f"`gem-rush-007.{team_key}_mlb_data_2024`"
        return f"`gem-rush-007.unknown_team_mlb_data_2024`"

    team_id = TEAMS[team_name]
    table_name = _get_table_name(team_name)

    try:
        query = f"""
        SELECT
            p.play_id,
            p.inning,
            p.half_inning,
            p.event,
            p.event_type,
            p.description,
            p.start_time,
            g.official_date as game_date
        FROM
            {table_name}.plays AS p
        INNER JOIN
            {table_name}.games AS g
            ON p.game_id = g.game_id
        INNER JOIN
            {table_name}.roster AS r
            ON p.batter_id = r.player_id
        WHERE
            r.full_name = @player_name
            AND (g.home_team_id = {team_id} OR g.away_team_id = {team_id})
            AND ((g.home_team_name = @opponent_team) OR (g.away_team_name = @opponent_team))
        """
        if specific_date:
            query += f" AND g.official_date = @specific_date AND DATE(p.start_time) = @specific_date"

        query += f"""
        ORDER BY
            p.end_time DESC,
            p.start_time ASC
        LIMIT @limit
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("player_name", "STRING", player_name),
                bigquery.ScalarQueryParameter("opponent_team", "STRING", opponent_team),
                bigquery.ScalarQueryParameter("limit", "INT64", limit),
                bigquery.ScalarQueryParameter("specific_date", "DATE", specific_date)
            ]
        )
        bq_client = bigquery.Client()
        query_job = bq_client.query(query, job_config=job_config)
        results = list(query_job.result())

        formatted_results = []
        for row in results:
            row_dict = dict(row)
            if 'start_time' in row_dict and row_dict['start_time'] and isinstance(row_dict['start_time'], datetime):
                row_dict['start_time'] = row_dict['start_time'].isoformat()
            if 'game_date' in row_dict and row_dict['game_date'] and isinstance(row_dict['game_date'], datetime.date):
                row_dict['game_date'] = row_dict['game_date'].isoformat()
            formatted_results.append(row_dict)
        return formatted_results

    except Exception as e:
        error_message = f"Error fetching player plays by opponent for {player_name} on {team_name}: {e}"
        logging.error(error_message)
        return [{"error": "Data error", "message": error_message}]




def fetch_player_plays_by_game_type(player_name: str, team_name: str, game_type: str, limit: int = 100, specific_date: str = None) -> list:
    """
    Fetches play-by-play data for a player in games of a specific type, optionally filtered by date.

    Args:
        player_name: Player's full name.
        team_name: Team name.
        game_type: Type of game (R, P, etc.).
        limit: Max number of plays.
        specific_date:  A specific date in 'YYYY-MM-DD' format.

    Returns:
        A list of dictionaries, each containing play details, or an error object.
    """
    from google.cloud import bigquery
    from google.cloud.bigquery import exceptions
    import logging
    from datetime import datetime

    logging.basicConfig(level=logging.INFO)

    TEAMS = {
        'rangers': 140,
        'angels': 108,
        'astros': 117,
        'rays': 139,
        'blue_jays': 141,
        'yankees': 147,
        'orioles': 110,
        'red_sox': 111,
        'twins': 142,
        'white_sox': 145,
        'guardians': 114,
        'tigers': 116,
        'royals': 118,
        'padres': 135,
        'giants': 137,
        'diamondbacks': 109,
        'rockies': 115,
        'phillies': 143,
        'braves': 144,
        'marlins': 146,
        'nationals': 120,
        'mets': 121,
        'pirates': 134,
        'cardinals': 138,
        'brewers': 158,
        'cubs': 112,
        'reds': 113,
        'athletics': 133,
        'mariners': 136,
        'dodgers': 119,
    }
    FULL_TEAM_NAMES = {
        'texas rangers': 'rangers',
        'los angeles angels': 'angels',
        'houston astros': 'astros',
        'tampa bay rays': 'rays',
        'toronto blue jays': 'blue_jays',
        'new york yankees': 'yankees',
        'baltimore orioles': 'orioles',
        'boston red sox': 'red_sox',
        'minnesota twins': 'twins',
        'chicago white sox': 'white_sox',
        'cleveland guardians': 'guardians',
        'detroit tigers': 'tigers',
        'kansas city royals': 'royals',
        'san diego padres': 'padres',
        'san francisco giants': 'giants',
        'arizona diamondbacks': 'diamondbacks',
        'colorado rockies': 'rockies',
        'philadelphia phillies': 'phillies',
        'atlanta braves': 'braves',
        'miami marlins': 'marlins',
        'washington nationals': 'nationals',
        'new york mets': 'mets',
        'pittsburgh pirates': 'pirates',
        'st louis cardinals': 'cardinals',
        'milwaukee brewers': 'brewers',
        'chicago cubs': 'cubs',
        'cincinnati reds': 'reds',
        'oakland athletics': 'athletics',
        'seattle mariners': 'mariners',
        'los angeles dodgers': 'dodgers',
    }
    def _get_table_name(team_name: str) -> str:
        """Helper function to construct the table name."""
        cleaned_name = team_name.lower().strip()
        if cleaned_name in FULL_TEAM_NAMES:
            team_key = FULL_TEAM_NAMES[cleaned_name]
            return f"`gem-rush-007.{team_key}_mlb_data_2024`"
        for team_key in TEAMS:
            if team_key in cleaned_name:
                return f"`gem-rush-007.{team_key}_mlb_data_2024`"
        return f"`gem-rush-007.unknown_team_mlb_data_2024`"

    team_id = TEAMS[team_name]
    table_name = _get_table_name(team_name)

    try:
        query = f"""
        SELECT
            p.play_id,
            p.inning,
            p.half_inning,
            p.event,
            p.event_type,
            p.description,
            p.start_time,
            g.official_date as game_date
        FROM
            {table_name}.plays AS p
        INNER JOIN
            {table_name}.games AS g
            ON p.game_id = g.game_id
        INNER JOIN
            {table_name}.roster AS r
            ON p.batter_id = r.player_id
        WHERE
            r.full_name = @player_name
            AND (g.home_team_id = {team_id} OR g.away_team_id = {team_id})
            AND g.game_type = @game_type
        """
        if specific_date:
            query += f" AND g.official_date = @specific_date AND DATE(p.start_time) = @specific_date"

        query += f"""
        ORDER BY
            p.end_time DESC,
            p.start_time ASC
        LIMIT @limit
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("player_name", "STRING", player_name),
                bigquery.ScalarQueryParameter("game_type", "STRING", game_type),
                bigquery.ScalarQueryParameter("limit", "INT64", limit),
                bigquery.ScalarQueryParameter("specific_date", "DATE", specific_date)
            ]
        )
        bq_client = bigquery.Client()
        query_job = bq_client.query(query, job_config=job_config)
        results = list(query_job.result())

        formatted_results = []
        for row in results:
            row_dict = dict(row)
            if 'start_time' in row_dict and row_dict['start_time'] and isinstance(row_dict['start_time'], datetime):
                row_dict['start_time'] = row_dict['start_time'].isoformat()
            if 'game_date' in row_dict and row_dict['game_date'] and isinstance(row_dict['game_date'], datetime.date):
                row_dict['game_date'] = row_dict['game_date'].isoformat()
            formatted_results.append(row_dict)
        return formatted_results

    except Exception as e:
        error_message = f"Error fetching player plays by game type for {player_name} on {team_name}: {e}"
        logging.error(error_message)
        return [{"error": "Data error", "message": error_message}]


# --- Define Agent ---
agent = reasoning_engines.LangchainAgent(
    model=model,
    tools=[
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
        # ... include all your adapted data functions ...
    ],
    agent_executor_kwargs={"return_intermediate_steps": True},
)

# --- Test Locally (IMPORTANT!) ---
print("Testing locally...")
test_query = "What were the results of the last two Rangers games?"
local_response = agent.query(input=test_query)
print(f"Local Response: {local_response}")

# for local_chunk in agent.stream_query(input=test_query):
#   print(local_chunk)


# --- Deploy to Vertex AI ---
print("Deploying to Vertex AI...")
remote_agent = reasoning_engines.ReasoningEngine.create(
    agent,
    requirements=[
        "google-cloud-aiplatform[langchain,reasoningengine]",
        "cloudpickle==3.0.0",
        "pydantic>=2.10",
        "requests",
        "google-cloud-bigquery",
        "google-cloud-secret-manager",
        "google-cloud-logging",
        "pymongo",
        "urllib3",
        # List ALL your dependencies
    ],
)

print(f"Deployed Reasoning Engine: {remote_agent.resource_name}")

# --- Test Remotely ---
print("Testing remotely...")
remote_response = remote_agent.query(input=test_query)
print(f"Remote Response: {remote_response}")

# for remote_chunk in remote_agent.stream_query(input=test_query):
#   print(remote_chunk)

# --- Example of how to use the deployed agent from another script/notebook ---
# print("Example usage from another script/notebook:")
# print(f"""
# from vertexai.preview import reasoning_engines

# resource name = projects/1011675918473/locations/us-central1/reasoningEngines/2601796355038904320
# REASONING_ENGINE_RESOURCE_NAME = "{remote_agent.resource_name}"  # Use the resource name

# remote_agent = reasoning_engines.ReasoningEngine(REASONING_ENGINE_RESOURCE_NAME)
# response = remote_agent.query(input="Give me a summary of the last Yankees game.")
# print(response)
# """)

# --- Clean Up (when done) ---
# remote_agent.delete() # Uncomment when ready to delete

# --- generate_mlb_podcasts (Example usage) ---
# Now, in a separate part of your notebook (or in a different script), you would use
# your `generate_mlb_podcasts` function, which interacts with the *deployed* agent.
# You do NOT deploy `generate_mlb_podcasts` itself.
#
# from your_app_module import generate_mlb_podcasts  # Assuming it's in your_app_module.py

# podcast_script = generate_mlb_podcasts("Create a podcast script about the last Rangers game.")
# print(podcast_script)