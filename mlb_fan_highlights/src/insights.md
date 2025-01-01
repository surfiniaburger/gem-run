 Identify players with unusually high batting averages compared to their slugging percentages, potentially indicating an anomaly in hitting power.

 ```bash
SELECT
  first_name,
  last_name,
  batting_average,
  slugging_percentage
FROM
  `mlb_data.combined_player_stats`
WHERE
  season BETWEEN 2020
  AND 2022
  AND batting_average > (
  SELECT
    AVG(batting_average) + 2 * STDDEV(batting_average)
  FROM
    `mlb_data.combined_player_stats`
  WHERE
    season BETWEEN 2020
    AND 2022 )
  AND slugging_percentage < (
  SELECT
    AVG(slugging_percentage) - 2 * STDDEV(slugging_percentage)
  FROM
    `mlb_data.combined_player_stats`
  WHERE
    season BETWEEN 2020
    AND 2022 );
 ```


Find players who have an unusually high number of stolen bases compared to the league average for their position.
```bash
WITH
  PlayerStolenBases AS (
  SELECT
    player_id,
    first_name,
    last_name,
    position_name,
    SUM(stolen_bases) AS total_stolen_bases
  FROM
    `mlb_data.combined_player_stats`
  WHERE
    season BETWEEN 2020
    AND 2022
  GROUP BY
    1,
    2,
    3,
    4 ),
  AvgStolenBasesByPosition AS (
  SELECT
    position_name,
    AVG(total_stolen_bases) AS avg_stolen_bases
  FROM
    PlayerStolenBases
  GROUP BY
    1 )
SELECT
  psb.first_name,
  psb.last_name,
  psb.position_name,
  psb.total_stolen_bases,
  asbp.avg_stolen_bases
FROM
  PlayerStolenBases psb
JOIN
  AvgStolenBasesByPosition asbp
ON
  psb.position_name = asbp.position_name
WHERE
  psb.total_stolen_bases > asbp.avg_stolen_bases * 2;
  ```


 Identify the team with the largest difference between their highest single-season win percentage and their lowest single-season win percentage.

```bash
WITH
  TeamWinPercentage AS (
  SELECT
    team_name,
    season,
    COUNTIF(home_score > away_score) * 100.0 / COUNT(*) AS win_percentage
  FROM
    `mlb_data.combined_player_stats`
  GROUP BY
    team_name,
    season )
SELECT
  team_name,
  MAX(win_percentage) - MIN(win_percentage) AS win_percentage_difference
FROM
  TeamWinPercentage
GROUP BY
  team_name
ORDER BY
  win_percentage_difference DESC
LIMIT
  1;
  ```


Calculate the percentage of players who have hit more home runs in away games compared to home games, considering only players who have hit at least 10 home runs in their career.

```bash
WITH
  PlayerHomeAwayHR AS (
  SELECT
    player_id,
    SUM(CASE
        WHEN home_team_id = team_id THEN homeruns
        ELSE 0
    END
      ) AS home_runs_home,
    SUM(CASE
        WHEN away_team_id = team_id THEN homeruns
        ELSE 0
    END
      ) AS home_runs_away
  FROM
    `mlb_data.combined_player_stats`
  GROUP BY
    player_id
  HAVING
    SUM(homeruns) >= 10 )
SELECT
  COUNTIF(home_runs_away > home_runs_home) * 100.0 / COUNT(*) AS percentage_away_hr_higher
FROM
  PlayerHomeAwayHR;
```

 Identify the player with the highest single-season batting average who has hit at least 30 home runs and stolen at least 20 bases in that season.

```bash
SELECT
  first_name,
  last_name,
  batting_average,
  season
FROM
  `mlb_data.combined_player_stats`
WHERE
  homeruns >= 30
  AND stolen_bases >= 20
ORDER BY
  batting_average DESC
LIMIT
  1;
```

Calculate the 3-game moving average of runs scored by each team for each season.

```bash
SELECT
  team_name,
  season,
  game_date,
  runs,
  AVG(runs) OVER (PARTITION BY team_name, season ORDER BY game_date ASC ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS moving_average_runs
FROM
  `mlb_data.combined_player_stats`
WHERE
  season BETWEEN 2010
  AND 2020
ORDER BY
  team_name,
  season,
  game_date;
```

This query identifies the player with the highest number of home runs in each season, along with their team name, by joining the table with itself.

```bash
SELECT
  t1.season,
  t1.first_name,
  t1.last_name,
  t1.team_name,
  t1.homeruns
FROM
  `mlb_data.combined_player_stats` AS t1
INNER JOIN (
  SELECT
    season,
    MAX(homeruns) AS max_homeruns
  FROM
    `mlb_data.combined_player_stats`
  GROUP BY
    season) AS t2
ON
  t1.season = t2.season
  AND t1.homeruns = t2.max_homeruns
```



This query calculates the 90th percentile of on-base plus slugging (OPS) for players in each position, considering only players who have played at least 50 games in a season, using APPROX_QUANTILES.
```bash
SELECT
  position_name,
  APPROX_QUANTILES(on_base_plus_slugging, 100)[
OFFSET
  (90)] AS ops_90th_percentile
FROM
  `mlb_data.combined_player_stats`
WHERE
  games_played >= 50
GROUP BY
  position_name
```



This query calculates the 90th percentile of on-base plus slugging (OPS) for players in each season using APPROX_QUANTILES.

```bash

SELECT
  season,
  APPROX_QUANTILES(on_base_plus_slugging, 100)[
OFFSET
  (90)] AS ops_90th_percentile
FROM
  `mlb_data.combined_player_stats`
GROUP BY
  season
  ```


Find the top 5 seasons with the highest average number of runs scored per game, considering both home and away teams.
SELECT
  season,
  AVG((home_score + away_score) / 2) AS avg_runs_per_game
FROM
  `mlb_data.combined_player_stats`
GROUP BY
  season
ORDER BY
  avg_runs_per_game DESC
LIMIT
  5;


Find the top 5 players with the highest strikeout-to-walk ratios across their careers.
```bash
SELECT
  first_name,
  last_name,
  SUM(strikeouts) / SUM(walks) AS strikeout_to_walk_ratio
FROM
  `mlb_data.combined_player_stats`
GROUP BY
  first_name,
  last_name
ORDER BY
  strikeout_to_walk_ratio DESC
LIMIT
  5;
```


Calculate the standard deviation of batting averages for players who debuted after the year 2000.
```bash
SELECT
  STDDEV(batting_average) AS stddev_batting_average
FROM
  `mlb_data.combined_player_stats`
WHERE
  CAST(first_year_of_play AS INT64) > 2000;
  ```


 Calculate the average number of stolen bases per game for each team in each season, considering only games where the home team won.
```bash
SELECT
  season,
  team_name,
  AVG(CAST(stolen_bases AS FLOAT64)) / COUNT(DISTINCT game_pk) AS avg_stolen_bases_per_game
FROM
  `mlb_data.combined_player_stats`
WHERE
  home_score > away_score
GROUP BY
  season,
  team_name;
```



This query finds the player with the highest on-base percentage for each team, considering only players with at least 300 at-bats, by grouping by team and player and then selecting the maximum OBP.

SELECT
  team_name,
  first_name,
  last_name,
  MAX(on_base_percentage) AS max_obp
FROM
  `mlb_data.combined_player_stats`
WHERE
  at_bats >= 300
GROUP BY
  team_name,
  first_name,
  last_name

Calculate the correlation between a team's total home runs and their total wins for each season.

WITH
  TeamWins AS (
  SELECT
    season,
    team_id,
    COUNTIF(home_score > away_score) AS wins
  FROM
    `mlb_data.combined_player_stats`
  WHERE
    season BETWEEN 2000
    AND 2022
  GROUP BY
    1,
    2 ),
  TeamHomeruns AS (
  SELECT
    season,
    team_id,
    SUM(homeruns) AS total_homeruns
  FROM
    `mlb_data.combined_player_stats`
  WHERE
    season BETWEEN 2000
    AND 2022
  GROUP BY
    1,
    2 )
SELECT
  tw.season,
  CORR(th.total_homeruns, tw.wins) AS homerun_win_correlation
FROM
  TeamWins AS tw
JOIN
  TeamHomeruns AS th
ON
  tw.season = th.season
  AND tw.team_id = th.team_id
GROUP BY
  1;



Find the correlation between a player's weight and the number of home runs they hit.
```bash
SELECT
  CORR(weight, homeruns) AS weight_homerun_correlation
FROM
  `mlb_data.combined_player_stats`
WHERE
  season BETWEEN 2000
  AND 2022;
  ```




Calculate the standard deviation of runs scored by each team in home games.
```bash
SELECT
  team_name,
  STDDEV_POP(home_score) AS stddev_home_runs
FROM
  `mlb_data.combined_player_stats`
WHERE
  season BETWEEN 2000
  AND 2022
GROUP BY
  1;
```




Find the average slugging percentage for players grouped by position, only considering players who have played more than 100 games.
```bash
SELECT
  position_name,
  AVG(slugging_percentage) AS average_slugging_percentage
FROM
  `mlb_data.combined_player_stats`
WHERE
  games_played > 100
GROUP BY
  position_name;
```



Identify the team with the highest average player weight in each season.

```bash
SELECT
  season,
  team_name,
  AVG(weight) AS average_player_weight
FROM
  `mlb_data.combined_player_stats`
GROUP BY
  season,
  team_name
ORDER BY
  average_player_weight DESC
LIMIT
  1;
```

 Calculate the percentage of games won by the home team for each season.
 ```bash
SELECT
  season,
  COUNTIF(home_score > away_score) * 100.0 / COUNT(*) AS home_win_percentage
FROM
  `mlb_data.combined_player_stats`
GROUP BY
  season;
  ```

Find the average number of strikeouts per game for each season, considering both home and away teams.
```bash
SELECT
  season,
  AVG(strikeouts) AS avg_strikeouts_per_game
FROM
  `mlb_data.combined_player_stats`
WHERE
  season BETWEEN 2000
  AND 2022
GROUP BY
  1;
```

Identify the player with the highest on-base plus slugging (OPS) for each season.
```bash
SELECT
  season,
  first_name,
  last_name,
  on_base_plus_slugging
FROM
  `mlb_data.combined_player_stats`
WHERE
  season BETWEEN 2000
  AND 2022
QUALIFY
  ROW_NUMBER() OVER (PARTITION BY season ORDER BY on_base_plus_slugging DESC) = 1;
  ```

Find the average number of games played by players in each position type (e.g., pitcher, catcher, infielder, outfielder).


```bash
SELECT
  position_type,
  AVG(games_played) AS avg_games_played
FROM
  `mlb_data.combined_player_stats`
WHERE
  season BETWEEN 2000
  AND 2022
GROUP BY
  1;
  ```


Calculate the monthly average home runs for each team over the years.
```bash
SELECT
  team_name,
  EXTRACT(MONTH
  FROM
    game_date) AS month,
  AVG(homeruns) AS average_home_runs
FROM
  `mlb_data.combined_player_stats`
WHERE
  game_date BETWEEN '2010-01-01'
  AND '2020-12-31'
GROUP BY
  team_name,
  month
ORDER BY
  team_name,
  month;
  ```


Calculate the average number of strikeouts per game for each team, considering both home and away games, for a specific season.

```bash
SELECT
  team_name,
  AVG(total_strikeouts) AS average_strikeouts_per_game
FROM (
  SELECT
    team_name,
    game_pk,
    SUM(strikeouts) AS total_strikeouts
  FROM
    `mlb_data.combined_player_stats`
  WHERE
    season = 2019
  GROUP BY
    team_name,
    game_pk )
GROUP BY
  team_name;
  ```


Determine the average number of runs scored by home teams in games where the away team scored more than 5 runs.

```bash
SELECT
  AVG(home_score) AS average_home_score
FROM
  `mlb_data.combined_player_stats`
WHERE
  away_score > 5;
  ```


 Determine the correlation between a player's height (in inches) and their total number of home runs hit.

```bash
SELECT
  CORR(CAST(REPLACE(height, '-', '.') AS FLOAT64) * 12, homeruns) AS height_homerun_correlation
FROM
  `mlb_data.combined_player_stats`;
```


`````````````````````````````````````
## Further Insights - Team and Player Performance Deep Dive

Here are some additional SQL queries to gain deeper insights into team and individual player performance, focusing on recent seasons (2023 and 2024 where applicable) and leveraging the `mlb_data.combined_player_stats` table.

**Player Focus - Exceptional Performances & Slumps:**

* **Closest to a Cycle (Hitting a Single, Double, Triple, and Home Run in one game):**


This query identifies instances where a player came close to hitting for the cycle, ordering the results by the most recent games first. This helps highlight near-exceptional offensive performances.
```sql
SELECT
    first_name,
    last_name,
    game_date,
    team_name
  FROM
    `mlb_data.combined_player_stats`
  WHERE season IN (2023, 2024)
    AND singles > 0
    AND doubles > 0
    AND triples > 0
    AND homeruns >0
ORDER BY game_date DESC


- This helps understand a team's home and away win records and their overall winning percentage across seasons, providing a comprehensive overview of performance trends.
WITH TeamWins AS (
  SELECT
    season,
    team_name,
    COUNTIF(home_score > away_score OR away_score > home_score) AS total_games,
    COUNTIF(home_score > away_score) AS home_wins,
    COUNTIF(away_score > home_score) AS away_wins
  FROM
    `mlb_data.combined_player_stats`
  WHERE season IN (2023, 2024)
  GROUP BY 1, 2
)
SELECT
  *,
  (home_wins + away_wins) * 100.0 / total_games AS win_percentage
FROM TeamWins
ORDER BY season, win_percentage DESC


This query aims to identify "unsung heroes" – players with high Wins Above Replacement (WAR) but perhaps less media attention. Adjust the WAR threshold as needed
SELECT
    first_name,
    last_name,
    team_name,
    season,
    wins_above_replacement  -- Assumes a WAR column exists
  FROM
    `mlb_data.combined_player_stats`
  WHERE season IN (2023, 2024)
   AND wins_above_replacement > 2.5  -- Example threshold
ORDER BY wins_above_replacement DESC




This query looks at the consistency of a team's run production on a game-by-game basis. The rolling standard deviation of runs scored can highlight periods of high variability (inconsistent scoring) versus more consistent offensive output.


SELECT
    team_name,
    game_date,
    home_score + away_score AS total_runs,
    STDDEV(home_score + away_score) OVER (PARTITION BY team_name ORDER BY game_date) AS rolling_stddev_runs
  FROM
    `mlb_data.combined_player_stats`
  WHERE season IN (2023, 2024)
ORDER BY team_name, game_date



##  Further Insights - Team and Player Performance Deep Dive

Here are some additional SQL queries to gain deeper insights into team and individual player performance, focusing on recent seasons (2023 and 2024 where applicable) and leveraging the `mlb_data.combined_player_stats` table.

**Player Focus - Exceptional Performances & Slumps:**

* **Best Single-Game Performance (based on a composite score):**

This query creates a composite performance_score based on various offensive statistics, allowing you to identify a player's best single-game performance. You can adjust the weighting of each statistic within the score as needed.
```sql
SELECT
    first_name,
    last_name,
    game_date,
    team_name,
    hits + doubles*2 + triples*3 + homeruns*4 + rbi + stolen_bases AS performance_score
  FROM
    `mlb_data.combined_player_stats`
  WHERE season IN (2023, 2024) -- Filter for recent seasons if applicable
ORDER BY
  performance_score DESC
LIMIT 1



This query compares a player's On-Base Plus Slugging (OPS) in one season versus the previous season to identify players with the greatest improvement. Adjust the WHERE clause to change the seasons being compared.
SELECT
    t1.first_name,
    t1.last_name,
    t1.team_name,
    t1.season,
    (t1.on_base_plus_slugging - t2.on_base_plus_slugging) AS ops_improvement
  FROM
    `mlb_data.combined_player_stats` AS t1
    INNER JOIN `mlb_data.combined_player_stats` AS t2 ON t1.player_id = t2.player_id
       AND t1.season = t2.season + 1  -- Compare to the previous season
  WHERE t1.season IN (2024)  -- Example: Comparing 2024 to 2023
ORDER BY ops_improvement DESC



-----------------------------AWS--------------------


-- 1. Player Hot Streak Analysis (2023)
-- Find players with the best consecutive game hitting streaks
WITH player_games AS (
  SELECT 
    full_name,
    game_date,
    hits,
    at_bats,
    batting_average,
    LAG(game_date) OVER (PARTITION BY player_id ORDER BY game_date) as prev_game_date
  FROM `mlb_data.combined_player_stats`
  WHERE season = 2023 
    AND at_bats > 0
)
SELECT 
    full_name,
    COUNT(*) as streak_length,
    AVG(batting_average) as avg_during_streak,
    MIN(game_date) as streak_start,
    MAX(game_date) as streak_end
FROM player_games
WHERE hits > 0
GROUP BY full_name, player_id
HAVING COUNT(*) >= 10
ORDER BY streak_length DESC, avg_during_streak DESC
LIMIT 10;

-- 2. Power Hitters Performance Analysis
SELECT 
    full_name,
    position_name,
    SUM(games_played) as total_games,
    SUM(homeruns) as total_homeruns,
    SUM(doubles) as total_doubles,
    SUM(triples) as total_triples,
    ROUND(AVG(slugging_percentage), 3) as avg_slugging,
    ROUND(AVG(on_base_plus_slugging), 3) as avg_ops
FROM `mlb_data.combined_player_stats`
WHERE season = 2023
GROUP BY full_name, position_name
HAVING total_games >= 50
ORDER BY total_homeruns DESC
LIMIT 15;

-- 3. Team Home vs Away Performance
SELECT 
    t.team_name,
    COUNT(CASE WHEN home_team_id = t.team_id THEN 1 END) as home_games,
    AVG(CASE WHEN home_team_id = t.team_id THEN home_score END) as avg_home_runs,
    COUNT(CASE WHEN away_team_id = t.team_id THEN 1 END) as away_games,
    AVG(CASE WHEN away_team_id = t.team_id THEN away_score END) as avg_away_runs
FROM `mlb_data.combined_player_stats` p
JOIN (SELECT DISTINCT team_id, team_name FROM `mlb_data.combined_player_stats`) t
ON p.team_id = t.team_id
WHERE season = 2023
GROUP BY t.team_name, t.team_id
HAVING home_games > 0 AND away_games > 0
ORDER BY (avg_home_runs - avg_away_runs) DESC;

-- 4. Most Improved Players (2022 vs 2023)
WITH player_stats_by_year AS (
  SELECT 
    full_name,
    position_name,
    season,
    SUM(games_played) as games,
    AVG(batting_average) as avg_batting,
    AVG(on_base_percentage) as avg_obp,
    AVG(slugging_percentage) as avg_slg
  FROM `mlb_data.combined_player_stats`
  WHERE season IN (2022, 2023)
  GROUP BY full_name, position_name, season
)
SELECT 
    p23.full_name,
    p23.position_name,
    ROUND(p23.avg_batting - p22.avg_batting, 3) as batting_improvement,
    ROUND(p23.avg_obp - p22.avg_obp, 3) as obp_improvement,
    ROUND(p23.avg_slg - p22.avg_slg, 3) as slg_improvement
FROM player_stats_by_year p23
JOIN player_stats_by_year p22 
  ON p23.full_name = p22.full_name 
  AND p23.season = 2023 
  AND p22.season = 2022
WHERE p23.games >= 50 AND p22.games >= 50
ORDER BY batting_improvement DESC
LIMIT 10;

-- 5. Speed Demons - Stolen Base Success Rate
SELECT 
    full_name,
    team_name,
    SUM(stolen_bases) as total_stolen_bases,
    SUM(caught_stealing) as times_caught,
    ROUND(SUM(stolen_bases) * 100.0 / NULLIF(SUM(stolen_bases + caught_stealing), 0), 2) as success_rate,
    SUM(games_played) as games_played
FROM `mlb_data.combined_player_stats`
WHERE season = 2023
GROUP BY full_name, team_name
HAVING total_stolen_bases >= 10
ORDER BY success_rate DESC, total_stolen_bases DESC
LIMIT 10;

--------------------------------------



Why this is a good idea:

Personalization: Focusing on the user's team or specific players makes the content directly relevant and interesting.

Engagement: The "good cop/bad cop" dynamic naturally creates conflict and different perspectives, making the podcast more entertaining and thought-provoking. It prevents the podcast from being a dry recitation of stats.

Insightful Analysis: Leveraging the data analysis capabilities of surfire.py ensures that the podcast has factual grounding and provides genuine insights beyond just opinions.

Unique Offering: This combination offers a unique way for fans to interact with data and analysis about their teams, going beyond traditional articles or highlight reels.


<iframe width="600" height="450" src="https://lookerstudio.google.com/embed/reporting/57ebdcdb-9526-44d3-9e47-4d01994f6f1c/page/eiCbE" frameborder="0" style="border:0" allowfullscreen sandbox="allow-storage-access-by-user-activation allow-scripts allow-same-origin allow-popups allow-popups-to-escape-sandbox"></iframe>

Looker Studio Report

































































































































































