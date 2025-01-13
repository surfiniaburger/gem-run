def fetch_player_plays(player_name: str, limit: int = 100) -> dict:
    LOOKER_STUDIO_BASE_URL = "https://lookerstudio.google.com/embed/reporting/f60f900b-9d43-46b8-b46a-4fba57e7637e/page/p_jsdpfv6qod"  # Replace with your actual base report URL
    """
    Fetches play-by-play data for a specific player from Dodgers games and generates a Looker Studio iframe URL.

    Args:
        player_name (str): Full name of the player
        limit (int, optional): Maximum number of plays to return. Defaults to 100.

    Returns:
        dict: A dictionary containing:
            - iframe_url (str): Looker Studio iframe URL with a filter for the player name
            - plays (list): List of dictionaries containing play-by-play data
    """
    try:
        query = """
        SELECT
            p.play_id,
            p.inning,
            p.half_inning,
            p.event,
            p.event_type,
            p.description,
            p.start_time,
            g.official_date as game_date,
         
        FROM
            `gem-rush-007.dodgers_mlb_data_2024.plays` AS p
        INNER JOIN 
            `gem-rush-007.dodgers_mlb_data_2024.games` AS g 
            ON p.game_id = g.game_id
        INNER JOIN
            `gem-rush-007.dodgers_mlb_data_2024.roster` AS r
            ON (p.batter_id = r.player_id OR p.pitcher_id = r.player_id)
        
        WHERE
            r.full_name = @player_name
            AND (g.home_team_id = 119 OR g.away_team_id = 119)
        ORDER BY 
            g.official_date DESC,
            p.start_time ASC
        LIMIT @limit
        """

        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("player_name", "STRING", player_name),
                bigquery.ScalarQueryParameter("limit", "INT64", limit)
            ]
        )
        
        query_job = bq_client.query(query, job_config=job_config)
        results = list(query_job.result())

        # Construct the Looker Studio iframe URL with a filter for the player name
        params = {
            "ds6": f"player_name_{player_name}",  # Replace 'df1' with the actual filter ID in your Looker Studio report, this is an example.
        }
        encoded_params = urllib.parse.urlencode(params)
        iframe_url = f"{LOOKER_STUDIO_BASE_URL}?{encoded_params}"
        

        # Convert the results to dictionaries and format datetime objects
        formatted_results = []
        for row in results:
            row_dict = dict(row)
            # Convert datetime objects to ISO format strings
            if 'start_time' in row_dict and row_dict['start_time']:
                row_dict['start_time'] = row_dict['start_time'].isoformat()
            if 'game_date' in row_dict and row_dict['game_date']:
                row_dict['game_date'] = row_dict['game_date'].isoformat()
            formatted_results.append(row_dict)
        return {
            "iframe_url": iframe_url,
            "plays": formatted_results
        }

    except Exception as e:
        logging.error(f"Error in fetch_player_plays: {e}")
        return []