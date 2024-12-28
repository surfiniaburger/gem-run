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





Calculate the average number of at-bats per hit for players in each position, considering only players who have had at least 200 at-bats in their career.
```bash
SELECT
  position_name,
  SUM(at_bats) / SUM(hits) AS avg_at_bats_per_hit
FROM
  `mlb_data.combined_player_stats`
WHERE
  at_bats >= 200
GROUP BY
  position_name;
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















































































































































































