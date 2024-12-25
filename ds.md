#Setup
---
#@title Import Python Libraries
# General data science libraries
import pandas as pd
import numpy as np

# Pulling data from APIs, parsing JSON
import requests
import json

# Interfacing w/ Cloud Storage from Python
from google.cloud import storage

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

from IPython.display import HTML
#@title Modify Settings

# Expand max column width when displaying data frames to handle longer text
pd.set_option('display.max_colwidth', 200)
# Exploring MLB Stats API
---
The MLB Stats API ([see documentation here](https://console.cloud.google.com/storage/browser/gcp-mlb-hackathon-2025/datasets/mlb-statsapi-docs)) provides access to various MLB historical and present day data feeds, without authentication. The various data feeds here should be very useful in making progress on this project. The cells below show how to access data from various MLB Stats API endpoints and do some initial processing.
#@title Function to Process Results from Various MLB Stats API Endpoints
def process_endpoint_url(endpoint_url, pop_key=None):
  """
  Fetches data from a URL, parses JSON, and optionally pops a key.

  Args:
    endpoint_url: The URL to fetch data from.
    pop_key: The key to pop from the JSON data (optional, defaults to None).

  Returns:
    A pandas DataFrame containing the processed data
  """
  json_result = requests.get(endpoint_url).content

  data = json.loads(json_result)

   # if pop_key is provided, pop key and normalize nested fields
  if pop_key:
    df_result = pd.json_normalize(data.pop(pop_key), sep = '_')
  # if pop_key is not provided, normalize entire json
  else:
    df_result = pd.json_normalize(data)

  return df_result
#@title Sports (Different Baseball Leagues/Levels/Competitions)
sports_endpoint_url = 'https://statsapi.mlb.com/api/v1/sports'

sports = process_endpoint_url(sports_endpoint_url, 'sports')

display(sports)

response ----
	id	code	link	name	abbreviation	sortOrder	activeStatus
0	1	mlb	/api/v1/sports/1	Major League Baseball	MLB	11	True
1	11	aaa	/api/v1/sports/11	Triple-A	AAA	101	True
2	12	aax	/api/v1/sports/12	Double-A	AA	201	True
3	13	afa	/api/v1/sports/13	High-A	A+	301	True
4	14	afx	/api/v1/sports/14	Single-A	A	401	True
5	16	rok	/api/v1/sports/16	Rookie	ROK	701	True
6	17	
7
8
... ... Output is truncated.

#@title Leagues

# Can add "?sportId=1" to following URL for MLB only
leagues_endpoint_url = 'https://statsapi.mlb.com/api/v1/league'

leagues = process_endpoint_url(leagues_endpoint_url, 'leagues')

display(leagues)

response-
id	name	link	abbreviation	nameShort	seasonState	hasWildCard	hasSplitSeason	numGames	hasPlayoffPoints	...	seasonDateInfo_postSeasonEndDate	seasonDateInfo_seasonEndDate	seasonDateInfo_offseasonStartDate	seasonDateInfo_offSeasonEndDate	seasonDateInfo_seasonLevelGamedayType	seasonDateInfo_gameLevelGamedayType	seasonDateInfo_qualifierPlateAppearances	seasonDateInfo_qualifierOutsPitched	sport_id	sport_link
0	103	American League	/api/v1/league/103	AL	American	offseason	True	False	162.0	False	...	2024-10-30	2024-10-30	2024-10-31	2024-12-31	P	P	3.1	3.0	1.0	/api/v1/sports/1
1	104	National League	/api/v1/league/104	NL	National	offseason	True	False	162.0	False	...	2024-10-30	2024-10-30	2024-10-31	2024-12-31	P	P	3.1	3.0	1.0	/api/v1/sports/1
2	114	Cactus League	/api/v1/league/114	CL	Cactus	offseason	False	False	NaN	False	...	NaN	NaN	2024-03-27	2024-12-31	F	F	NaN	NaN	NaN	NaN
3	115	Grapefruit League	/api/v1/league/115	GL	Grapefruit	offseason	False	False	NaN	False	...	NaN	NaN	2024-03-27	2024-12-31	F	F	NaN	NaN	NaN	NaN
4	117	International League	/api/v1/league/117	INT	International	offseason	True	True	150.0	False	...	2024-09-28	2024-09-28	2024-09-29	2024-12-31	Y	Y	2.7	2.4	11.0	/api/v1/sports/11
...	...	Output is truncated.

#@title Seasons

# Use "?sportId=1" in following URL for MLB only
# Can also add "&withGameTypeDates=true" at end to get much more info on games
seasons_endpoint_url = 'https://statsapi.mlb.com/api/v1/seasons/all?sportId=1'

seasons = process_endpoint_url(seasons_endpoint_url, 'seasons')

display(seasons)

	seasonId	hasWildcard	preSeasonStartDate	seasonStartDate	regularSeasonStartDate	regularSeasonEndDate	seasonEndDate	offseasonStartDate	offSeasonEndDate	seasonLevelGamedayType	...	qualifierPlateAppearances	qualifierOutsPitched	postSeasonStartDate	postSeasonEndDate	lastDate1stHalf	allStarDate	firstDate2ndHalf	preSeasonEndDate	springStartDate	springEndDate
0	1876	False	1876-01-01	1876-04-22	1876-04-22	1876-10-09	1876-10-09	1876-10-10	1877-04-29	S	...	3.1	3.0	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN
1	1877	False	1877-01-01	1877-04-30	1877-04-30	1877-10-06	1877-10-06	1877-10-07	1878-04-30	S	...	3.1	3.0	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN
2	1878	False	1878-01-01	1878-05-01	1878-05-01	1878-09-30	1878-09-30	1878-10-01	1879-04-30	S	...	3.1	3.0	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN
3	1879	False	1879-01-01	1879-05-01	1879-05-01	1879-09-30	1879-09-30	1879-10-01	1880-04-30	S	...	3.1	3.0	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN
4	1880
... ... ... ...Output is truncated.



#@title Teams
# Use "?sportId=1" in following URL for MLB only
teams_endpoint_url = 'https://statsapi.mlb.com/api/v1/teams?sportId=1'

teams = process_endpoint_url(teams_endpoint_url, 'teams')

display(teams)

response - 
	allStarStatus	id	name	link	season	teamCode	fileCode	abbreviation	teamName	locationName	...	springVenue_link	league_id	league_name	league_link	division_id	division_name	division_link	sport_id	sport_link	sport_name
0	N	133	Athletics	/api/v1/teams/133	2024	ath	ath	ATH	Athletics	Sacramento	...	/api/v1/venues/2507	103	American League	/api/v1/league/103	200	American League West	/api/v1/divisions/200	1	/api/v1/sports/1	Major League Baseball
1	N	134	Pittsburgh Pirates	/api/v1/teams/134	2024	pit	pit	PIT	Pirates	Pittsburgh	...	/api/v1/venues/2526	104	National League	/api/v1/league/104	205	National League Central	/api/v1/divisions/205	1	/api/v1/sports/1	Major League Baseball
2	N	135	San Diego Padres	/api/v1/teams/135	2024	sdn	sd	SD	Padres	San Diego	...	/api/v1/venues/2530	104	National League	/api/v1/league/104	203	National League West	/api/v1/divisions/203	1	/api/v1/sports/1	Major League Baseball
3	... ... ...
... ...Output is truncated.

#@title Single Team Roster

# Pick single team ID to get roster for (default is 119 for Dodgers)
team_id = 119 # @param {type:"integer"}

single_team_roster_url = f'https://statsapi.mlb.com/api/v1/teams/{team_id}/roster?season=2024'

single_team_roster = process_endpoint_url(single_team_roster_url, 'roster')

display(single_team_roster)

	jerseyNumber	parentTeamId	person_id	person_fullName	person_link	position_code	position_name	position_type	position_abbreviation	status_code	status_description
0	51	119	681911	Alex Vesia	/api/v1/people/681911	1	Pitcher	Pitcher	P	A	Active
1	44	119	681624	Andy Pages	/api/v1/people/681624	8	Outfielder	Outfielder	CF	A	Active
2	43	119	607455	Anthony Banda	/api/v1/people/607455	1	Pitcher	Pitcher	P	A	Active
3	15	119	605131	Austin Barnes	/api/v1/people/605131	2	Catcher	Catcher	C	A	Active
4	78	119	676508	Ben Casparius	/api/v1/people/676508	1	Pitcher	Pitcher	P	A	Active
5	7 ... ...
...Output is truncated.

#@title Single Player Information
# Pick single player ID to get info for (default is 660271 for Shohei Ohtani)
player_id = 660271 # @param {type:"integer"}

single_player_url = f'https://statsapi.mlb.com/api/v1/people/{player_id}/'

single_player_info_json = json.loads(requests.get(single_player_url).content)

display(single_player_info_json)

{'copyright': 'Copyright 2024 MLB Advanced Media, L.P.  Use of any content on this page acknowledges agreement to the terms posted here http://gdx.mlb.com/components/copyright.txt',
 'people': [{'id': 660271,
   'fullName': 'Shohei Ohtani',
   'link': '/api/v1/people/660271',
   'firstName': 'Shohei',
   'lastName': 'Ohtani',
   'primaryNumber': '17',
   'birthDate': '1994-07-05',
   'currentAge': 30,
   'birthCity': 'Oshu',
   'birthCountry': 'Japan',
   'height': '6\' 4"',
   'weight': 210,
   'active': True,
   'primaryPosition': {'code': 'Y',
    'name': 'Two-Way Player',
    'type': 'Two-Way Player',
    'abbreviation': 'TWP'},
   'useName': 'Shohei',
   'useLastName': 'Ohtani',
   'boxscoreName': 'Ohtani',
   'nickName': 'Showtime',
   'gender': 'M',
   'isPlayer': True,
   'isVerified': False,
...
   'initLastName': 'S Ohtani',
   'fullFMLName': 'Shohei Ohtani',
   'fullLFMName': 'Ohtani, Shohei',
   'strikeZoneTop': 3.4,
   'strikeZoneBottom': 1.62}]}
Output is truncated. Vi


#@title Schedule / Games
# Use "?sportId=1" in following URL for MLB only
# Can change season to get other seasons' games info
schedule_endpoint_url = 'https://statsapi.mlb.com/api/v1/schedule?sportId=1&season=2024'

schedule_dates = process_endpoint_url(schedule_endpoint_url, "dates")

games = pd.json_normalize(
    schedule_dates.explode('games').reset_index(drop = True)['games'])

display(games)

response -
	gamePk	gameGuid	link	gameType	season	gameDate	officialDate	isTie	gameNumber	publicFacing	...	status.reason	description	rescheduleDate	rescheduleGameDate	rescheduledFrom	rescheduledFromDate	resumeDate	resumeGameDate	resumedFrom	resumedFromDate
0	748266	d5cb4300-04fc-4cd0-9a62-88099e61bd81	/api/v1.1/game/748266/feed/live	S	2024	2024-02-22T20:10:00Z	2024-02-22	False	1	True	...	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN
1	748344	1ae43208-ddd5-4d32-af13-334857bddb80	/api/v1.1/game/748344/feed/live	E	2024	2024-02-23T18:05:00Z	2024-02-23	False	1	True	...	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN
2	748263	33b22841-8ae2-4589-9e93-64bfe9efdf8a	/api/v1.1/game/748263/feed/live	S	2024	2024-02-23T20:05:00Z	2024-02-23	False	1	True	...	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN
3	748341	1e162b85-6a35-4a0b-a0f1-da6ee9b74b4d	/api/v1.1/game/748341/feed/live	S	2024	2024-02-23T20:05:00Z	2024-02-23	False	1	True	...	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	Na
output truncated

After getting the full season schedule, we can pick 1 game (via "gamePk") to pull detailed data for, as is done below (we default to the last game in the result above).
#@title Single Game Full Data

# Pick gamePK of last game from games data as default
game_pk = games['gamePk'].iloc[-1]

single_game_feed_url = f'https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live'

single_game_info_json = json.loads(requests.get(single_game_feed_url).content)

# Print the initial part of the JSON result (very large object with many fields)
display(json.dumps(single_game_info_json)[:1000])

response -
'{"copyright": "Copyright 2024 MLB Advanced Media, L.P.  Use of any content on this page acknowledges agreement to the terms posted here http://gdx.mlb.com/components/copyright.txt", "gamePk": 775296, "link": "/api/v1.1/game/775296/feed/live", "metaData": {"wait": 10, "timeStamp": "20241031_035122", "gameEvents": ["strikeout", "game_finished"], "logicalEvents": ["midInning", "countChange", "count13", "gameStateChangeToGameOver"]}, "gameData": {"game": {"pk": 775296, "type": "W", "doubleHeader": "N", "id": "2024/10/30/lanmlb-nyamlb-1", "gamedayType": "P", "tiebreaker": "N", "gameNumber": 1, "calendarEventID": "14-775296-2024-10-30", "season": "2024", "seasonDisplay": "2024"}, "datetime": {"dateTime": "2024-10-31T00:08:00Z", "originalDate": "2024-10-30", "officialDate": "2024-10-30", "dayNight": "night", "time": "8:08", "ampm": "PM"}, "status": {"abstractGameState": "Final", "codedGameState": "F", "detailedState": "Final", "statusCode": "F", "startTimeTBD": false, "abstractGameCode": "F"}'
---

That game data feed has a lot of detailed information about the game itself, the teams, the players, and what happened on every pitch. Below, we extract all the information on the last ("current") play from the game chosen above, to show all the information available for every pitch.
#@title Single Play Information (from Game Data)
# Default to getting info on "current" (last) play from single game info above
single_game_play = single_game_info_json['liveData']['plays']['currentPlay']

display(single_game_play)

{'result': {'type': 'atBat',
  'event': 'Strikeout',
  'eventType': 'strikeout',
  'description': 'Alex Verdugo strikes out swinging.',
  'rbi': 0,
  'awayScore': 7,
  'homeScore': 6,
  'isOut': True},
 'about': {'atBatIndex': 88,
  'halfInning': 'bottom',
  'isTopInning': False,
  'inning': 9,
  'startTime': '2024-10-31T03:50:09.726Z',
  'endTime': '2024-10-31T03:51:22.288Z',
  'isComplete': True,
  'isScoringPlay': False,
  'hasReview': False,
  'hasOut': True,
  'captivatingIndex': 14},
 'count': {'balls': 1, 'strikes': 3, 'outs': 3},
 'matchup': {'batter': {'id': 657077,
   'fullName': 'Alex Verdugo',
   'link': '/api/v1/people/657077'},
  'batSide': {'code': 'L', 'description': 'Left'},
  'pitcher': {'id': 621111,
...
   'endTime': '2024-10-31T03:51:22.288Z',
   'isPitch': True,
   'type': 'pitch'}],
 'playEndTime': '2024-10-31T03:51:22.288Z',
 'atBatIndex': 88}
Output is truncated. V

 [MLB Film Room](https://www.mlb.com/video) gives fans incredible access to watch, create and share baseball highlights and videos from the game. The cell below shows how to take a single MLB playId from MLB Stats API (like the ones available in some of the outputs above) and then build a URL to find the video for that play on Film Room.
#@title Get MLB Film Room Video Link for Specific Play ID
# Pick single play ID to get info for (default is Freddie Freeman 2024 WS Gm1 walk-off grand slam)
play_id = "560a2f9b-9589-4e4b-95f5-2ef796334a94" # @param {type:"string"}

single_play_video_url = f'https://www.mlb.com/video/search?q=playid=\"{play_id}\"'

display(single_play_video_url)

response -
'https://www.mlb.com/video/search?q=playid="560a2f9b-9589-4e4b-95f5-2ef796334a94"'
---

# Exploring 2024 MLB Home Runs Data and Video
---
[One of the provided datasets](https://storage.googleapis.com/gcp-mlb-hackathon-2025/datasets/2024-mlb-homeruns.csv) has a link to a public video file for each home run hit during the 2024 MLB regular season, along with some basic information about every HR. These links can be used to watch video to help corroborate what's in the data for a specific HR, and also the video can serve as a basis for AI-driven analysis or recommendations.
#@title Get 2024 MLB Home Runs Data from Cloud Storage

mlb_2024_hr_csv = 'https://storage.googleapis.com/gcp-mlb-hackathon-2025/datasets/2024-mlb-homeruns.csv'

mlb_2024_hr_df = pd.read_csv(mlb_2024_hr_csv)

display(mlb_2024_hr_df.head())

play_id	title	ExitVelocity	HitDistance	LaunchAngle	video
0	148e943b-10db-4d71-943d-ead3b36bebbc	Freddie Freeman homers (1) on a fly ball to center field. Shohei Ohtani scores.	101.7	408.819710	33.0	https://sporty-clips.mlb.com/eVozQWVfWGw0TUFRPT1fQndWWkFWMEFWVkFBQ1ZKV0JBQUFWUUZYQUZnQlVBVUFWd1JSQTFFR0IxRUFVbEFG.mp4
1	d4116f91-e362-4261-a7d0-fc02f0eeb67b	Mookie Betts homers (2) on a fly ball to left field.	104.6	406.399234	28.0	https://sporty-clips.mlb.com/eVozQWVfWGw0TUFRPT1fVWdWUVZWSlNYd01BWFZFRVZBQUFBd0JYQUFBQ0IxRUFWQVlGQWxVRlZGWUhCZ1VF.mp4
2	2d6363ad-8ff2-49cc-bf21-d8cdfdc55999	George Spring
--- output truncated
To render 1 of the public .mp4 videos within the notebook, provide the URL and run the cell below.
#@title See Single Home Run Video in Notebook

# Pick 1 MLB video URL to see video (default is Shohei Ohtani's 50th HR of 2024)
video_url = 'https://sporty-clips.mlb.com/TVpSTTVfWGw0TUFRPT1fQmdGV1VWUlhWMUFBWEZaVEJ3QUFWUTVYQUZoWFdsY0FWMTBEQndzREFGWUJCbGNF.mp4' #@param {type:"string"}

HTML(f"""<video width="640" height="360" controls>
          <source src="{video_url}" type="video/mp4">
          Your browser does not support the video tag.
        </video>""")
# Exploring MLB Fan Favorites Data
---
[One of the provided datasets](https://storage.googleapis.com/gcp-mlb-hackathon-2025/datasets/mlb-fan-content-interaction-data/2025-mlb-fan-favs-follows-000000000000.json) has information about fans' favorite teams (1 team per fan) and followed teams (potentially multiple teams per fan) from MLB digital properties.
#@title Function to Load Newline Delimited JSON into Pandas DF
def load_newline_delimited_json(url):
    """Loads a newline-delimited JSON file from a URL into a pandas DataFrame.

    Args:
        url: The URL of the newline-delimited JSON file.

    Returns:
        A pandas DataFrame containing the data, or None if an error occurs.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes

        data = []
        for line in response.text.strip().split('\n'):
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Skipping invalid JSON line: {line} due to error: {e}")

        return pd.DataFrame(data)
    except requests.exceptions.RequestException as e:
        print(f"Error downloading data: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None
#@title Read in MLB Fan Favorites/Follows Data from Google Cloud Storage
mlb_fan_favorites_json_file = 'https://storage.googleapis.com/gcp-mlb-hackathon-2025/datasets/mlb-fan-content-interaction-data/2025-mlb-fan-favs-follows-000000000000.json'

mlb_fan_favorites_df = load_newline_delimited_json(mlb_fan_favorites_json_file)

# Convert favorite team ID to integer format
mlb_fan_favorites_df['favorite_team_id'] = (
  mlb_fan_favorites_df['favorite_team_id'].astype('Int64'))

display(mlb_fan_favorites_df.head())

response - 

	user_id	favorite_team_id	followed_team_ids
0	ECAMUCPLNDAWAT8	108	[119, 114, 143, 142, 110, 108]
1	YHL4PR5KKW8H0KS	108	[147, 109, 119, 136, 139, 144, 140, 135]
2	TBN0IX5JK19LIB7	108	[112, 119, 147, 144, 137]
3	DXYBMAV0B040YC5	108	[116, 119]
4	UEJ33ZL4CLZTG7P	108	[121, 137, 147, 119]
--- ---

#@title Look at Most Common Favorite MLB Teams
most_common_favorite_teams = (pd.merge(
  mlb_fan_favorites_df['favorite_team_id'].value_counts().reset_index().
    rename(columns = {"count": "num_favorites"}),
  teams[['id', 'name']].
    rename(columns = {"id": "team_id", "name": "team_name"}),
  left_on = 'favorite_team_id',
  right_on = 'team_id',
  how = 'left'
  )[['team_id', 'team_name', 'num_favorites']]
  )

# Create barplot showing most common favorite MLB teams
plt.figure(figsize=(12, 8))
sns.barplot(x='num_favorites', y='team_name', data=most_common_favorite_teams,
    orient='h', color='blue')
plt.title('Most Common Favorite MLB Teams')
plt.xlabel('Number of Favorites')
plt.ylabel('Team Name')

# Add text labels for # of favorites next to each bar
for index, row in most_common_favorite_teams.iterrows():
  plt.text(row['num_favorites'], index, str(row['num_favorites']),
    color='black', ha='left', va='center')

plt.show()
#@title Look at Most Followed MLB Teams

# Explode the 'followed_team_ids' column to create 1 row for each followed team
mlb_fan_follows_expanded_df = (mlb_fan_favorites_df.
  explode('followed_team_ids').
  reset_index(drop=True)
  )

# Convert followed team IDs to integer format
mlb_fan_follows_expanded_df['followed_team_ids'] = (
  mlb_fan_follows_expanded_df['followed_team_ids'].astype('Int64'))

most_followed_teams = (pd.merge(
  mlb_fan_follows_expanded_df['followed_team_ids'].value_counts().reset_index().
    rename(columns = {"count": "num_followers"}),
  teams[['id', 'name']].
    rename(columns = {"id": "team_id", "name": "team_name"}),
  left_on = 'followed_team_ids',
  right_on = 'team_id',
  how = 'left'
  )[['team_id', 'team_name', 'num_followers']]
  )

# Create barplot showing most followed MLB teams
plt.figure(figsize=(12, 8))
sns.barplot(x='num_followers', y='team_name', data=most_followed_teams,
    orient='h', color='blue')
plt.title('Most Followed MLB Teams')
plt.xlabel('Number of Followers')
plt.ylabel('Team Name')

# Add text labels for # of followers next to each bar
for index, row in most_followed_teams.iterrows():
  plt.text(row['num_followers'], index, str(row['num_followers']),
    color='black', ha='left', va='center')

plt.show()
# Exploring MLB Fan Content Interaction Data
---
There is [a large dataset](https://console.cloud.google.com/storage/browser/gcp-mlb-hackathon-2025/datasets/mlb-fan-content-interaction-data) with data on the interaction of MLB fans with various content on MLB digital properties. Below, we read in just 1 of the dozens of JSON files with this information in to show what it looks like.
#@title Read in Example MLB Fan Content Interaction Data File from Google Cloud Storage
mlb_fan_content_interaction_json_file = 'https://storage.googleapis.com/gcp-mlb-hackathon-2025/datasets/mlb-fan-content-interaction-data/mlb-fan-content-interaction-data-000000000000.json'

mlb_fan_content_interaction_df = load_newline_delimited_json(
    mlb_fan_content_interaction_json_file)

display(mlb_fan_content_interaction_df)


response -
date_time_date	date_time_utc	source	user_id	slug	content_type	content_headline	team_ids	player_tags
0	2024-07-30	2024-07-31T03:21:21+00:00	iOS	AZNYWKP68NTBLWN	rodriguez-charles-combine-to-turn-great-double-play	video	Rodriguez, Charles combine to turn great double play	[]	[691497, 800051]
1	2024-07-30	2024-07-30T18:57:53+00:00	iOS	WZO387IQWPX4HTL	chris-okey-hits-a-walk-off-single	video	Chris Okey hits a walk-off single	[]	[608360]
2	2024-07-30	2024-07-30T23:05:11+00:00	iOS	JIOSQ23LVKD7RHO	chris-okey-hits-a-walk-off-single	video	Chris Okey hits a walk-off single	[]	[608360]
3	2024-07-30	
---- --output truncated

#@title See What Dates, Content Types, and Sources Are Present
date_counts = mlb_fan_content_interaction_df['date_time_date'].value_counts()

display(date_counts)

content_type_counts = (mlb_fan_content_interaction_df['content_type'].
    value_counts())

display(content_type_counts)


content_source_counts = (mlb_fan_content_interaction_df['source'].
    value_counts())

display(content_source_counts)

response -
	count
date_time_date	
2024-07-30	648664
dtype: int64
count
content_type	
video	465867
article	182797
dtype: int64
count
source	
iOS	353133
Web	224632
Android	70899
--- ---

#@title Find Content Pieces with Most Interaction in This Data
interaction_by_content = (mlb_fan_content_interaction_df[
    ['slug', 'content_type', 'content_headline']].
    value_counts().
    reset_index().
    rename(columns = {"count": "num_interactions"})
    )

display(interaction_by_content)

response -

	slug	content_type	content_headline	num_interactions
0	every-2024-mlb-trade-deadline-deal	article	Tracking every 2024 Trade Deadline deal	35181
1	c-2523586283	video	Gameday Video Placement clip	19273
2	jazz-chisholm-jr-homers-17-on-a-fly-ball-to-right-field-juan-soto-scores	video	Jazz Chisholm Jr.'s second homer of the day (17)	17492
3	aaron-nola-in-play-run-s-to-jazz-chisholm-jr	video	Jazz Chisholm Jr.'s RBI groundout	11859
4	mlb-ru
--- --- output truncated

#@title Generate MLB.com Link for Article or Video for Specific Content Piece
# Pick single content piece to get link for
content_slug = "every-2024-mlb-trade-deadline-deal" # @param {type:"string"}
content_type = "article" # @param {type:"string"} ['article', 'video']

content_type_cat = ('news' if (content_type == 'article') else 'video')

content_mlb_com_link = f'https://www.mlb.com/{content_type_cat}/{content_slug}'

print(content_mlb_com_link)

response -
https://www.mlb.com/news/every-2024-mlb-trade-deadline-deal
---

# Exploring MLB Caption Data
---

There is [an interesting text-heavy dataset](https://console.cloud.google.com/storage/browser/gcp-mlb-hackathon-2025/datasets/mlb-caption-data) with captions from the game broadcast of some MLB games, mapped to timestamps within the game (so that you can match play-level data with these captions). Below, we read in all of the JSON files with this information in, combine them into 1 data frame, and do some preliminary analysis.
#@title Read in All MLB Caption Data from Google Cloud Storage
mlb_captions_base_url = 'https://storage.googleapis.com/gcp-mlb-hackathon-2025/datasets/mlb-caption-data/mlb-captions-data-*.json'
all_dfs = []
i = 0

# Loop over files labeled ""...00" to "...12"
for i in np.arange(0, 13):
    this_url = mlb_captions_base_url.replace("*", str(i).zfill(12))
    this_df = load_newline_delimited_json(this_url)
    all_dfs.append(this_df)
    i += 1

mlb_captions_df = pd.concat(all_dfs, ignore_index=True)

display(mlb_captions_df.head())

response -
caption_start	caption_end	caption_text	write_date	game_pk	feed_type
0	02:13:28	02:13:42.610000	And part of my thinking here, Steven Vogt leaving him in is not so much the fifth inning that may be sacrificing this matchup in the fifth, but perhaps it will be a righty when he comes around aga...	2024-09-20T07:32:20.420407+00:00	746579	A
1	04:04:23.465000	04:04:23.630000	So.	2024-09-20T07:32:20.420407+00:00	746579	A
2	03:16:07.572000	03:16:07.750000	100.	2024-09-20T07:32:20.420407+00:00	746579	A
3	01:08:21.947000	01:08:22.09000
----- output truncated


#@title See What Dates and Feed Types Are Present
# Convert 'write_date' to date only field
mlb_captions_df['write_date_only'] = (pd.to_datetime(
    mlb_captions_df['write_date']).dt.date)

date_counts = mlb_captions_df['write_date_only'].value_counts()

display(date_counts)

feed_type_counts = mlb_captions_df['feed_type'].value_counts()

display(feed_type_counts)

response -

	count
write_date_only	
2024-09-19	72721
2024-09-22	67778
2024-09-21	58176
2024-09-26	53861
2024-09-18	50628
2024-09-29	44870
2024-09-25	41066
2024-09-27	34961
2024-09-17	27606
2024-09-20	22291
2024-09-23	21038
2024-09-24	15361
2024-09-28	12669
2024-09-30	9863
2024-10-01	7246
dtype: int64
count
feed_type	
H	271118
A	269017
dtype: int64
--- --- 

#@title Get MLB Film Room Video Clip for Last Play from Specific Game
# Pick game to get last play from (default is game_pk 747066, for Braves-Royals
# game with Travis d'Arnaud walk-off HR on 9/28/2024)
game_pk = '747066' #@param{type:"string"}

single_game_feed_url = f'https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live'

single_game_info_json = json.loads(requests.get(single_game_feed_url).content)

single_game_play = single_game_info_json['liveData']['plays']['currentPlay']

single_game_play_id = single_game_play['playEvents'][-1]['playId']

single_play_video_url = f'https://www.mlb.com/video/search?q=playid=\"{single_game_play_id}\"'

display(single_play_video_url)


response -
'https://www.mlb.com/video/search?q=playid="ba04fcba-8fea-4f49-8768-6dc9230bbbe5"'
---- ----

#@title Get Captions Data Corresponding to Specific Play


# This is specific to the walk-off HR in game_pk 747066, with caption time codes
# found manually
single_play_captions = (mlb_captions_df[
    (mlb_captions_df['game_pk'] == game_pk)
    &
    (mlb_captions_df['feed_type'] == 'H')
    &
    (mlb_captions_df['caption_start'] >= '03:08:25.00000')
    &
    (mlb_captions_df['caption_end'] <= '03:10:21.0000')
    ].
    sort_values(['caption_start']).
    reset_index(drop = True)
    )

display(single_play_captions[['caption_start', 'caption_end', 'caption_text']])

response -
caption_start	caption_end	caption_text
0	03:08:25.594000	03:08:38.740000	>> Now Travis Otano first time facing the Royals since 2016, Travis has not seen much of the Royals, but he's got a spohere in the ninthith one out.
1	03:08:38.740000	03:08:45.310000	body on where he can ctainly make an impact impact.
2	03:08:45.314000	03:09:10.130000	Downstairs, two balls, no strikes to the Braves catcher catcherDa catcherDat catcherDa catcherDar field.
3	03:09:10.138000	03:09:15.240000	It's deep and it's gone gone.
4	03:09:15.243000	03:09:19.580000	>> Travis Darnel with the biggest home run of the Braves season.
5	03:09:19.581000	03:09:25.350000	And they walk it off 2 to 1.
6	03:09:25.354000	03:09:31.360000	Now Now.
7	03:09:31.360000	03:10:03.220000	Larrondo Dalvy Dalvy.
8	03:10:03.225000	03:10:08.860000	Something is eling different about this team right now now.
9	03:10:08.864000	03:10:10.380000	>> And what we are seeing when it matters.
10	03:10:10.382000	03:10:12.060000	The most.
11	03:10:12.067000	03:10:20.940000	And what a moment for Travis Otano second whichend has hit a wa off homer, he's ahead in the count two zero.
--- ---
# Other Resources
---

Below are a few other resources related to the hackathon or MLB in general that may be useful.

*   [Google Cloud x MLB Hackathon Resources GitHub Repo](https://github.com/MajorLeagueBaseball/google-cloud-mlb-hackathon)
*   [MLB Glossary](https://www.mlb.com/glossary)
*   ["MLB-StatsAPI" Python library to wrap MLB Stats API](https://pypi.org/project/MLB-StatsAPI/)
*   ["baseballR" R library to wrap MLB Stats API](https://cran.r-project.org/web/packages/baseballr/index.html)


# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Using Gemini Function Calling to Get Real-Time Company News and Insights

<table align="left">
  <td style="text-align: center">
    <a href="https://colab.research.google.com/github/GoogleCloudPlatform/generative-ai/blob/main/gemini/function-calling/use_case_company_news_and_insights.ipynb">
      <img src="https://cloud.google.com/ml-engine/images/colab-logo-32px.png" alt="Google Colaboratory logo"><br> Open in Colab
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/colab/import/https:%2F%2Fraw.githubusercontent.com%2FGoogleCloudPlatform%2Fgenerative-ai%2Fmain%2Fgemini%2Ffunction-calling%2Fuse_case_company_news_and_insights.ipynb">
      <img width="32px" src="https://lh3.googleusercontent.com/JmcxdQi-qOpctIvWKgPtrzZdJJK-J3sWE1RsfjZNwshCFgE_9fULcNpuXYTilIR2hjwN" alt="Google Cloud Colab Enterprise logo"><br> Open in Colab Enterprise
    </a>
  </td>    
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/workbench/deploy-notebook?download_url=https://raw.githubusercontent.com/GoogleCloudPlatform/generative-ai/main/gemini/function-calling/use_case_company_news_and_insights.ipynb">
      <img src="https://lh3.googleusercontent.com/UiNooY4LUgW_oTvpsNhPpQzsstV5W8F7rYgxgGBD85cWJoLmrOzhVs_ksK_vgx40SHs7jCqkTkCk=e14-rj-sc0xffffff-h130-w32" alt="Vertex AI logo"><br> Open in Workbench
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/function-calling/use_case_company_news_and_insights.ipynb">
      <img src="https://cloud.google.com/ml-engine/images/github-logo-32px.png" alt="GitHub logo"><br> View on GitHub
    </a>
  </td>
</table>

<div style="clear: both;"></div>

<b>Share to:</b>

<a href="https://www.linkedin.com/sharing/share-offsite/?url=https%3A//github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/function-calling/use_case_company_news_and_insights.ipynb" target="_blank">
  <img width="20px" src="https://upload.wikimedia.org/wikipedia/commons/8/81/LinkedIn_icon.svg" alt="LinkedIn logo">
</a>

<a href="https://bsky.app/intent/compose?text=https%3A//github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/function-calling/use_case_company_news_and_insights.ipynb" target="_blank">
  <img width="20px" src="https://upload.wikimedia.org/wikipedia/commons/7/7a/Bluesky_Logo.svg" alt="Bluesky logo">
</a>

<a href="https://twitter.com/intent/tweet?url=https%3A//github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/function-calling/use_case_company_news_and_insights.ipynb" target="_blank">
  <img width="20px" src="https://upload.wikimedia.org/wikipedia/commons/5/53/X_logo_2023_original.svg" alt="X logo">
</a>

<a href="https://reddit.com/submit?url=https%3A//github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/function-calling/use_case_company_news_and_insights.ipynb" target="_blank">
  <img width="20px" src="https://redditinc.com/hubfs/Reddit%20Inc/Brand/Reddit_Logo.png" alt="Reddit logo">
</a>

<a href="https://www.facebook.com/sharer/sharer.php?u=https%3A//github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/function-calling/use_case_company_news_and_insights.ipynb" target="_blank">
  <img width="20px" src="https://upload.wikimedia.org/wikipedia/commons/5/51/Facebook_f_logo_%282019%29.svg" alt="Facebook logo">
</a>            
| | |
|-|-|
| Author(s) | [Ishana Shinde](https://github.com/ishana7), [Kristopher Overholt](https://github.com/koverholt) |
## Function Calling in Gemini

Gemini is a family of generative AI models developed by Google DeepMind that is designed for multimodal use cases. [Function Calling in Gemini](https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/function-calling) lets developers create a description of a function in their code, then pass that description to a language model in a request. The response from the model includes the name of a function that matches the description and the arguments to call it with.

## Overview

Meet Jane. She's a busy investor who's always on the lookout for the latest market trends and financial news. She needs information quickly and accurately, but sifting through endless articles and reports is time-consuming.

Jane discovers [Function Calling](https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/function-calling), which is a powerful tool that uses the Gemini model to predict function calls to external systems and synthesizes information in natural language. Now Jane can get instant insights on companies and breaking news, all within her familiar coding environment. She can even build a web app on top of these APIs and functions so that her coworkers can benefit from this approach without writing any code!
### Define functions and parameter descriptions

Define function declarations that will be used as tools for Gemini by specifying the function details as a dictionary in accordance with the [OpenAPI JSON schema](https://spec.openapis.org/oas/v3.0.3#schemawr).

You'll define four tools to fetch various company and financial information, including stock prices, company overviews, news for a given company, and news sentiment for a given topic:
get_stock_price = FunctionDeclaration(
    name="get_stock_price",
    description="Fetch the current stock price of a given company",
    parameters={
        "type": "object",
        "properties": {
            "ticker": {
                "type": "string",
                "description": "Stock ticker symbol for a company",
            }
        },
    },
)

get_company_overview = FunctionDeclaration(
    name="get_company_overview",
    description="Get company details and other financial data",
    parameters={
        "type": "object",
        "properties": {
            "ticker": {
                "type": "string",
                "description": "Stock ticker symbol for a company",
            }
        },
    },
)

get_company_news = FunctionDeclaration(
    name="get_company_news",
    description="Get the latest news headlines for a given company.",
    parameters={
        "type": "object",
        "properties": {
            "tickers": {
                "type": "string",
                "description": "Stock ticker symbol for a company",
            }
        },
    },
)

get_news_with_sentiment = FunctionDeclaration(
    name="get_news_with_sentiment",
    description="Gets live and historical market news and sentiment data",
    parameters={
        "type": "object",
        "properties": {
            "news_topic": {
                "type": "string",
                "description": """News topic to learn about. Supported topics
                               include blockchain, earnings, ipo,
                               mergers_and_acquisitions, financial_markets,
                               economy_fiscal, economy_monetary, economy_macro,
                               energy_transportation, finance, life_sciences,
                               manufacturing, real_estate, retail_wholesale,
                               and technology""",
            },
        },
    },
)
### Wrap function declarations in a tool

Now, you can define a tool that will allow Gemini to select from the set of functions we've defined:
company_insights_tool = Tool(
    function_declarations=[
        get_stock_price,
        get_company_overview,
        get_company_news,
        get_news_with_sentiment,
    ],
)
### Company and financial information API

Alpha Vantage provides real-time and historical financial market data through a set of data APIs. In this tutorial, you'll use the Alpha Vantage API to get stock prices, company information, and news about different industries.

You can register for a free developer API key at [Alpha Vantage](https://www.alphavantage.co/). Once you have an API key, paste it into the cell below:
# API key for company and financial information
API_KEY = "PASTE_YOUR_API_KEY_HERE"
You'll use this API key throughout the rest of this notebook to make API requests and get information about various companies and industries.
### Define Python functions and a function handler
Define Python functions that you'll invoke to fetch data from an external API:
def get_stock_price_from_api(content):
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={content['ticker']}&apikey={API_KEY}"
    api_request = requests.get(url)
    return api_request.text


def get_company_overview_from_api(content):
    url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={content['ticker']}&apikey={API_KEY}"
    api_response = requests.get(url)
    return api_response.text


def get_company_news_from_api(content):
    url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={content['tickers']}&limit=20&sort=RELEVANCE&apikey={API_KEY}"
    api_response = requests.get(url)
    return api_response.text


def get_news_with_sentiment_from_api(content):
    url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&topics={content['news_topic']}&limit=20&sort=RELEVANCE&apikey={API_KEY}"
    api_request = requests.get(url)
    return api_request.text
Define a function handler that maps function call names (from your function declarations) to actual Python functions that call APIs:
function_handler = {
    "get_stock_price": get_stock_price_from_api,
    "get_company_overview": get_company_overview_from_api,
    "get_company_news": get_company_news_from_api,
    "get_news_with_sentiment": get_news_with_sentiment_from_api,
}
### Initialize model

Initialize the Gemini model with the desired model parameters and `Tool` that we defined earlier:
gemini_model = GenerativeModel(
    "gemini-1.5-pro-002",
    generation_config=GenerationConfig(temperature=0),
    tools=[company_insights_tool],
)
### Initialize chat session
chat = gemini_model.start_chat()
### Define a helper function to send chat messages and handle function calls & responses

Before you start chatting with the Gemini model and making function calls, recall that Gemini Function Calling predicts a function call from a set of functions then returns structured information about which function to call and which parameters to use.

Rather than having to manually inspect the predicted function names and function parameters and then repeatedly invoking API calls, the following helper function automates the process of handling API calls and responses to and from the Gemini model:
def send_chat_message(prompt):
    display(Markdown("#### Prompt"))
    print(prompt, "\n")
    prompt += """
    Give a concise, high-level summary. Only use information that you learn from 
    the API responses. 
    """

    # Send a chat message to the Gemini API
    response = chat.send_message(prompt)

    # Handle cases with multiple chained function calls
    function_calling_in_process = True
    while function_calling_in_process:
        # Extract the function call response
        function_call = response.candidates[0].content.parts[0].function_call

        # Check for a function call or a natural language response
        if function_call.name in function_handler.keys():
            # Extract the function call name
            function_name = function_call.name
            display(Markdown("#### Predicted function name"))
            print(function_name, "\n")

            # Extract the function call parameters
            params = {key: value for key, value in function_call.args.items()}
            display(Markdown("#### Predicted function parameters"))
            print(params, "\n")

            # Invoke a function that calls an external API
            function_api_response = function_handler[function_name](params)[
                :20000
            ]  # Stay within the input token limit
            display(Markdown("#### API response"))
            print(function_api_response[:500], "...", "\n")

            # Send the API response back to Gemini, which will generate a natural language summary or another function call
            response = chat.send_message(
                Part.from_function_response(
                    name=function_name,
                    response={"content": function_api_response},
                ),
            )
        else:
            function_calling_in_process = False

    # Show the final natural language summary
    display(Markdown("#### Natural language response"))
    display(Markdown(response.text.replace("$", "\\\\$")))
In the above helper function, the `while` loop handles cases in which the Gemini model predicts two or more chained Function Calls. The code within the `if` statement handles the invocation of function calls and API requests and responses. And the line of code in the `else` statement stops the Function Calling logic in the event that Gemini generates a natural language summary.

### Ask questions about various companies and topics

Now that you've defined your functions, initialized the Gemini model, and started a chat session, you're ready to ask questions!

### Sample prompt related to stock price

Start with a simple prompt that asks about a stock price:
send_chat_message("What is the current stock price for Google?")
#### How it works

Nice work! The output includes a concise summary of the real-time stock price for Alphabet, Inc.

Let's walk through the step-by-step end process that your application code went through, from the input prompt to the output summary:

1. Gemini used information within your prompt and predicted the `get_stock_price()` function along with the ticker symbol `GOOG`.
1. Your helper function then invoked an API call to retrieve the latest stock ticker information about Alphabet Inc.
1. Once you returned the API response to Gemini, it used this information to generate a natural language summary with the stock price of Alphabet Inc.

### Sample prompt related to company information
send_chat_message("Give me a company overview of Google")
#### How it works

For this prompt, Gemini predicted the `get_company_overview()` function along with the ticker symbol `GOOG`. The logic within your helper function handled the API call, and the natural language response generated by Gemini includes information about financial metrics, a company description, and stock details.

### Sample prompt for information about multiple companies

Now, see what happens what you ask about two different companies:
send_chat_message("Give me a company overview of Walmart and The Home Depot")
#### How it works

Great! This time, Gemini predicted the use of two subsequent function calls to `get_company_overview()`, one for each ticker symbol. The logic within your helper function handled the chained function calls, and the natural language response generated by Gemini includes information about both companies.

### Sample prompt related to company news

Ask a question about the latest news related to a particular company:
send_chat_message("What's the latest news about Google?")
#### How it works

For this prompt, Gemini predicted the `get_company_news()` function. The logic within your helper function handled the API call, and the natural language response generated by Gemini includes the latest news related to Google.

### Sample prompt related to industry news

Now, try sending a prompt about news for a particular industry:
send_chat_message("Has there been any exciting news related to real estate recently?")
#### How it works

This time, Gemini predicted the `get_news_with_sentiment()` function along with the function parameter `real_estate` as defined in your `FunctionDeclaration`. The logic within your helper function handled the API call, and the natural language response generated by Gemini includes the latest news and sentiment in the real estate industry.

### Summary

This tutorial highlights how Gemini Function Calling helps bridge the gap between raw data and actionable insights. This functionality empowers users to ask questions in natural language, our application code makes API calls to retrieve the latest relevant information, then the Gemini model summarizes the results from one or more API calls.

We're excited to see how you'll use Gemini Function calling to build generative AI applications that can help users make informed decisions, whether they are investors like Jane, or anyone who's looking to combine the power of generative AI models with reliable and up-to-date information from external data sources.

Feel free to try sending additional prompts, editing the function declarations, or even adding your own. Happy Function Calling!

# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Multimodal Function Calling with the Gemini API & Python SDK

<table align="left">
  <td style="text-align: center">
    <a href="https://colab.research.google.com/github/GoogleCloudPlatform/generative-ai/blob/main/gemini/function-calling/multimodal_function_calling.ipynb">
      <img src="https://cloud.google.com/ml-engine/images/colab-logo-32px.png" alt="Google Colaboratory logo"><br> Run in Colab
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/colab/import/https:%2F%2Fraw.githubusercontent.com%2FGoogleCloudPlatform%2Fgenerative-ai%2Fmain%2Fgemini%2Ffunction-calling%2Fmultimodal_function_calling.ipynb">
      <img width="32px" src="https://lh3.googleusercontent.com/JmcxdQi-qOpctIvWKgPtrzZdJJK-J3sWE1RsfjZNwshCFgE_9fULcNpuXYTilIR2hjwN" alt="Google Cloud Colab Enterprise logo"><br> Run in Colab Enterprise
    </a>
  </td>      
  <td style="text-align: center">
    <a href="https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/function-calling/multimodal_function_calling.ipynb">
      <img src="https://cloud.google.com/ml-engine/images/github-logo-32px.png" alt="GitHub logo"><br> View on GitHub
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/workbench/deploy-notebook?download_url=https://raw.githubusercontent.com/GoogleCloudPlatform/generative-ai/main/gemini/function-calling/multimodal_function_calling.ipynb">
      <img src="https://lh3.googleusercontent.com/UiNooY4LUgW_oTvpsNhPpQzsstV5W8F7rYgxgGBD85cWJoLmrOzhVs_ksK_vgx40SHs7jCqkTkCk=e14-rj-sc0xffffff-h130-w32" alt="Vertex AI logo"><br> Open in Vertex AI Workbench
    </a>
  </td>
</table>

<div style="clear: both;"></div>

<b>Share to:</b>

<a href="https://www.linkedin.com/sharing/share-offsite/?url=https%3A//github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/function-calling/multimodal_function_calling.ipynb" target="_blank">
  <img width="20px" src="https://upload.wikimedia.org/wikipedia/commons/8/81/LinkedIn_icon.svg" alt="LinkedIn logo">
</a>

<a href="https://bsky.app/intent/compose?text=https%3A//github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/function-calling/multimodal_function_calling.ipynb" target="_blank">
  <img width="20px" src="https://upload.wikimedia.org/wikipedia/commons/7/7a/Bluesky_Logo.svg" alt="Bluesky logo">
</a>

<a href="https://twitter.com/intent/tweet?url=https%3A//github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/function-calling/multimodal_function_calling.ipynb" target="_blank">
  <img width="20px" src="https://upload.wikimedia.org/wikipedia/commons/5/53/X_logo_2023_original.svg" alt="X logo">
</a>

<a href="https://reddit.com/submit?url=https%3A//github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/function-calling/multimodal_function_calling.ipynb" target="_blank">
  <img width="20px" src="https://redditinc.com/hubfs/Reddit%20Inc/Brand/Reddit_Logo.png" alt="Reddit logo">
</a>

<a href="https://www.facebook.com/sharer/sharer.php?u=https%3A//github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/function-calling/multimodal_function_calling.ipynb" target="_blank">
  <img width="20px" src="https://upload.wikimedia.org/wikipedia/commons/5/51/Facebook_f_logo_%282019%29.svg" alt="Facebook logo">
</a>            

| | |
|-|-|
|Author(s) | [Kristopher Overholt](https://github.com/koverholt) |
## Overview

### Introduction to Multimodal Function Calling with Gemini

This notebook demonstrates a powerful [Function Calling](https://cloud.google.com/vertex-ai/docs/generative-ai/multimodal/function-calling) capability of the Gemini model: support for multimodal inputs. With multimodal function calling, you can go beyond traditional text inputs, enabling Gemini to understand your intent and predict function calls and function parameters based on various inputs like images, audio, video, and PDFs. Function calling can also be referred to as *function calling with controlled generation*, which guarantees that output generated by the model always adheres to a specific schema so that you receive consistently formatted responses.

Previously, implementing multimodal function calling required two separate calls to the Gemini API: one to extract information from media, and another to generate a function call based on the extracted text. This process was cumbersome, prone to errors, and resulted in the loss of detail in valuable contextual information. Gemini's multimodal function calling capability streamlines this workflow, enabling a single API call that efficiently processes multimodal inputs for accurate function predictions and structured outputs. 

### How It Works

1. **Define Functions and Tools:** Describe your functions, then group them into `Tool` objects for Gemini to use.
2. **Send Inputs and Prompt:** Provide Gemini with multimodal input (image, audio, PDF, etc.) and a prompt describing your request.
3. **Gemini Predicts Action:** Gemini analyzes the multimodal input and prompt to predict the best function to call and its parameters.
4. **Execute and Return:** Use Gemini's prediction to make API calls, then send the results back to Gemini.
5. **Generate Response:** Gemini uses the API results to provide a final, natural language response to the user. 

This notebook will guide you through practical examples of using Gemini's multimodal function calling to build intelligent applications that go beyond the limitations of text-only interactions. 
### Objectives

In this tutorial, you will learn how to use the Gemini API in Vertex AI with the Vertex AI SDK for Python to make function calls with multimodal inputs, using the Gemini 1.5 Pro (`gemini-1.5-pro`) model. You'll explore how Gemini can process and understand various input types — including images, video, audio, and PDFs — to predict and execute functions.

You will complete the following tasks:

- Install the Vertex AI SDK for Python.
- Define functions that can be called by Gemini.
- Package functions into tools.
- Send multimodal inputs (images, video, audio, PDFs) and prompts to Gemini.
- Extract predicted function calls and their parameters from Gemini's response.
- Use the predicted output to make API calls to external systems (demonstrated with an image input example). 
- Return API responses to Gemini for natural language response generation (demonstrated with an image input example). 
### Costs

This tutorial uses billable components of Google Cloud:

- Vertex AI

Learn about [Vertex AI pricing](https://cloud.google.com/vertex-ai/pricing) and use the [Pricing Calculator](https://cloud.google.com/products/calculator/) to generate a cost estimate based on your projected usage.

## Getting Started

### Install Vertex AI SDK for Python

%pip install --upgrade --user --quiet google-cloud-aiplatform wikipedia
### Restart current runtime

To use the newly installed packages in this Jupyter runtime, you must restart the runtime. You can do this by running the cell below, which will restart the current kernel.
# Restart kernel after installs so that your environment can access the new packages
import IPython

app = IPython.Application.instance()
app.kernel.do_shutdown(True)
<div class="alert alert-block alert-warning">
<b>⚠️ The kernel is going to restart. Please wait until it is finished before continuing to the next step. ⚠️</b>
</div>

### Authenticate your notebook environment (Colab only)

If you are running this notebook on Google Colab, run the following cell to authenticate your environment. This step is not required if you are using [Vertex AI Workbench](https://cloud.google.com/vertex-ai-workbench).
import sys

if "google.colab" in sys.modules:
    from google.colab import auth

    auth.authenticate_user()
### Set Google Cloud project information and initialize Vertex AI SDK

To get started using Vertex AI, you must have an existing Google Cloud project and [enable the Vertex AI API](https://console.cloud.google.com/flows/enableapi?apiid=aiplatform.googleapis.com).

Learn more about [setting up a project and a development environment](https://cloud.google.com/vertex-ai/docs/start/cloud-environment).
PROJECT_ID = "[your-project-id]"  # @param {type:"string"}
LOCATION = "us-central1"  # @param {type:"string"}

import vertexai

vertexai.init(project=PROJECT_ID, location=LOCATION)
## Multimodal Function Calling in Action
### Import libraries

from IPython.display import Markdown, display
from vertexai.generative_models import (
    Content,
    FunctionDeclaration,
    GenerationConfig,
    GenerativeModel,
    Part,
    Tool,
)
import wikipedia
### Image-Based Function Calling: Finding Animal Habitats

In this example, you'll send along an image of a bird and ask Gemini to identify its habitat. This involves defining a function that looks up regions where a given animal is found, creating a tool that uses this function, and then sending a request to Gemini.

<img src="https://storage.googleapis.com/github-repo/generative-ai/gemini/function-calling/multi-color-bird.jpg" width="250px">

First, you define a `FunctionDeclaration` called `get_wildlife_region`. This function takes the name of an animal species as input and returns information about its typical region.
get_wildlife_region = FunctionDeclaration(
    name="get_wildlife_region",
    description="Look up the region where an animal can be found",
    parameters={
        "type": "object",
        "properties": {
            "animal": {"type": "string", "description": "Species of animal"}
        },
    },
)
Next, you create a `Tool` object that includes your `get_wildlife_region` function. Tools help group related functions that Gemini can use:
image_tool = Tool(
    function_declarations=[
        get_wildlife_region,
    ],
)
Now you're ready to send a request to Gemini. Initialize the `GenerativeModel` and specify the image to analyze, along with a prompt. The `tools` argument tells Gemini to consider the functions in your `image_tool`.
model = GenerativeModel("gemini-1.5-pro")
generation_config = GenerationConfig(temperature=0)

response = model.generate_content(
    [
        Part.from_uri(
            "gs://github-repo/generative-ai/gemini/function-calling/multi-color-bird.jpg",
            mime_type="image/jpeg",
        ),
        "What is the typical habitat or region where this animal lives?",
    ],
    generation_config=generation_config,
    tools=[image_tool],
)
response_function_call = response.candidates[0].content
response.candidates[0].function_calls[0]
Let's examine the response from Gemini. You can extract the predicted function name:
function_name = response.candidates[0].function_calls[0].name
function_name
You can also get the arguments that Gemini predicted for the function call:
function_args = {
    key: value for key, value in response.candidates[0].function_calls[0].args.items()
}
function_args
Now, you'll call an external API (in this case, using the `wikipedia` Python package) using the animal name that Gemini extracted from the image:
api_response = wikipedia.page(function_args["animal"]).content
api_response[:500]
Finally, you return the API response to Gemini so it can generate a final answer in natural language:
response = model.generate_content(
    [
        Content(
            role="user",
            parts=[
                Part.from_uri(
                    "gs://github-repo/generative-ai/gemini/function-calling/multi-color-bird.jpg",
                    mime_type="image/jpeg",
                ),
                Part.from_text(
                    "Inspect the image and get the regions where this animal can be found",
                ),
            ],
        ),
        response_function_call,  # Function call response
        Content(
            parts=[
                Part.from_function_response(
                    name=function_name,
                    response={
                        "content": api_response,  # Return the API response to the Gemini model
                    },
                )
            ],
        ),
    ],
    tools=[image_tool],
)

display(Markdown(response.text))
This example showcases how Gemini's multimodal function calling processes an image, predicts a relevant function and its parameters, and integrates with external APIs to provide comprehensive user information. This process opens up exciting possibilities for building intelligent applications that can "see" and understand the world around them via API calls to Gemini.
### Video-Based Function Calling: Identifying Product Features
Now let's explore how Gemini can extract information from videos for the purpose of invoking a function call. You'll use a video showcasing multiple products and ask Gemini to identify its key features.

<img src="https://storage.googleapis.com/github-repo/generative-ai/gemini/function-calling/made-by-google-24.gif" width="600px">

Start by defining a function called `get_feature_info` that takes a list of product features as input and could potentially be used to retrieve additional details about those features:
get_feature_info = FunctionDeclaration(
    name="get_feature_info",
    description="Get additional information about a product feature",
    parameters={
        "type": "object",
        "properties": {
            "features": {
                "type": "array",
                "description": "A list of product features",
                "items": {"type": "string", "description": "Product feature"},
            }
        },
    },
)
Next, create a tool that includes your `get_feature_info` function:
video_tool = Tool(
    function_declarations=[
        get_feature_info,
    ],
)
Send a video to Gemini, along with a prompt asking for information about the product features, making sure to include your `video_tool` in the `tools` kwarg:
model = GenerativeModel("gemini-1.5-pro")
generation_config = GenerationConfig(temperature=0)

response = model.generate_content(
    [
        Part.from_uri(
            "gs://github-repo/generative-ai/gemini/function-calling/made-by-google-24.mp4",
            mime_type="video/mp4",
        ),
        "Inspect the video and get information about the product features shown",
    ],
    generation_config=generation_config,
    tools=[video_tool],
)
response.candidates[0].function_calls
Gemini correctly predicted the `get_feature_info` function:
function_name = response.candidates[0].function_calls[0].name
function_name
And you can see the list of product features that Gemini extracted from the video, which are available as structured function arguments that adhere to the JSON schema we defined in the `FunctionDeclaration`:
function_args = {
    key: value for key, value in response.candidates[0].function_calls[0].args.items()
}
function_args
This example demonstrates Gemini's ability to understand video content. By defining a relevant function, you can use Gemini to extract structured information from videos and perform further actions based on that information.

Now that the multimodal function call response is complete, you could use the function name and function arguments to call an external API using any REST API or client library of your choice, similar to how we did in the previous example with the `wikipedia` Python package.

Since this sample notebook is focused on the mechanics of multimodal function calling rather than the subsequent function calls and API calls, we'll move on to another example with different multimodal inputs. You can refer to other sample notebooks on Gemini Function Calling for more details on where to go from here.
### Audio-Based Function Calling: Generating Book Recommendations
In this example, you'll explore using audio input with Gemini's multimodal function calling. You'll send a podcast episode to Gemini and ask for book recommendations related to the topics discussed.

<font color="green">>>> "SRE is just a production system specific manifestation of systems thinking ... and we kind of do it in an informal way."</font><br/>
<font color="purple">>>> "The book called 'Thinking in Systems' ... it's a really good primer on this topic."</font><br/>
<font color="green">>>> "An example of ... systems structure behavior thinking ... is the idea of like the cascading failure, that kind of vicious cycle of load that causes retries that causes more load ... "</font><br/>
<font color="purple">>>> "The worst pattern is the single embedded SRE that turns into the ops person ... you just end up doing all of the toil, all of the grunt work."</font><br/>
<font color="green">>>> "Take that moment, take a breath, and really analyze the problem and understand how it's working as a system and understand how you can intervene to improve that."</font><br/>
<font color="purple">>>> "Avoid just doing what you've done before and kicking the can down the road, and really think deeply about your problems."</font><br/>

Define a function called `get_recommended_books` that takes a list of topics as input and (hypothetically) returns relevant book recommendations:
get_recommended_books = FunctionDeclaration(
    name="get_recommended_books",
    description="Get recommended books based on a list of topics",
    parameters={
        "type": "object",
        "properties": {
            "topics": {
                "type": "array",
                "description": "A list of topics",
                "items": {"type": "string", "description": "Topic"},
            },
        },
    },
)
Now create a tool that includes your newly defined function:
audio_tool = Tool(
    function_declarations=[
        get_recommended_books,
    ],
)
Provide Gemini with the audio file and a prompt to recommend books based on the podcast content:
model = GenerativeModel("gemini-1.5-pro")
generation_config = GenerationConfig(temperature=0)

response = model.generate_content(
    [
        Part.from_uri(
            "gs://github-repo/generative-ai/gemini/function-calling/google-cloud-sre-podcast-s2-e8.mp3",
            mime_type="audio/mpeg",
        ),
        "Inspect the audio file and generate a list of recommended books based on the topics discussed",
    ],
    generation_config=generation_config,
    tools=[audio_tool],
)
response.candidates[0].function_calls
You can see that Gemini has successfully predicted your `get_recommended_books` function:
function_name = response.candidates[0].function_calls[0].name
function_name
And the function arguments contain the list of topics that Gemini identified and extracted from the input audio file:
function_args = {
    key: value for key, value in response.candidates[0].function_calls[0].args.items()
}
function_args
This example highlights Gemini's capacity to understand and extract information from audio, enabling you to create applications that respond to spoken content or audio-based interactions.
### PDF-Based Function Calling: Extracting Company Data from Invoices
This example demonstrates how to use Gemini's multimodal function calling to process PDF documents. You'll work with a set of invoices and extract the names of the (fictitious) companies involved.

<img src="https://storage.googleapis.com/github-repo/generative-ai/gemini/function-calling/invoice-synthetic-overview.png" width="1000px">

Define a function called `get_company_information` that (in a real-world scenario) could be used to fetch details about a given list of companies:
get_company_information = FunctionDeclaration(
    name="get_company_information",
    description="Get information about a list of companies",
    parameters={
        "type": "object",
        "properties": {
            "companies": {
                "type": "array",
                "description": "A list of companies",
                "items": {"type": "string", "description": "Company name"},
            }
        },
    },
)
Package your newly defined function into a tool:
invoice_tool = Tool(
    function_declarations=[
        get_company_information,
    ],
)
Now you can provide Gemini with multiple PDF invoices and ask it to get company information:
model = GenerativeModel("gemini-1.5-pro")
generation_config = GenerationConfig(temperature=0)

response = model.generate_content(
    [
        Part.from_uri(
            "gs://github-repo/generative-ai/gemini/function-calling/invoice-synthetic-1.pdf",
            mime_type="application/pdf",
        ),
        Part.from_uri(
            "gs://github-repo/generative-ai/gemini/function-calling/invoice-synthetic-2.pdf",
            mime_type="application/pdf",
        ),
        Part.from_uri(
            "gs://github-repo/generative-ai/gemini/function-calling/invoice-synthetic-3.pdf",
            mime_type="application/pdf",
        ),
        Part.from_uri(
            "gs://github-repo/generative-ai/gemini/function-calling/invoice-synthetic-4.pdf",
            mime_type="application/pdf",
        ),
        Part.from_uri(
            "gs://github-repo/generative-ai/gemini/function-calling/invoice-synthetic-5.pdf",
            mime_type="application/pdf",
        ),
        "Inspect the PDF files of invoices and retrieve information about each company",
    ],
    generation_config=generation_config,
    tools=[invoice_tool],
)
response.candidates[0].function_calls
As expected, Gemini predicted the `get_company_information` function:
function_name = response.candidates[0].function_calls[0].name
function_name
The function arguments contain the list of company names extracted from the PDF invoices:
function_args = {
    key: value for key, value in response.candidates[0].function_calls[0].args.items()
}
function_args
This example shows the power of Gemini for processing and extracting structured data from documents, a common requirement in many real-world applications.
### Image-Based Chat: Building a Multimodal Chatbot
Let's put it all together and build a simple multimodal chatbot. This chatbot will understand image inputs and respond to questions using the functions you define.

<img src="https://storage.googleapis.com/github-repo/generative-ai/gemini/function-calling/baby-fox-info.png" width="500px">

First, define three functions: `get_animal_details`, `get_location_details`, and `check_color_palette`. These functions represent the capabilities of your chatbot and could potentially be used to retrieve additional details using REST API calls:
get_animal_details = FunctionDeclaration(
    name="get_animal_details",
    description="Look up information about a given animal species",
    parameters={
        "type": "object",
        "properties": {
            "animal": {"type": "string", "description": "Species of animal"}
        },
    },
)

get_location_details = FunctionDeclaration(
    name="get_location_details",
    description="Look up information about a given location",
    parameters={
        "type": "object",
        "properties": {"location": {"type": "string", "description": "Location"}},
    },
)

check_color_palette = FunctionDeclaration(
    name="check_color_palette",
    description="Check hex color codes for accessibility",
    parameters={
        "type": "object",
        "properties": {
            "colors": {
                "type": "array",
                "description": "A list of colors in hexadecimal format",
                "items": {
                    "type": "string",
                    "description": "Hexadecimal representation of color, as in #355E3B",
                },
            }
        },
    },
)
Group your functions into a tool:
chat_tool = Tool(
    function_declarations=[
        get_animal_details,
        get_location_details,
        check_color_palette,
    ],
)
Initialize the `GenerativeModel` and start a chat session with Gemini, providing it with your `chat_tool`:
model = GenerativeModel(
    "gemini-1.5-pro",
    generation_config=GenerationConfig(temperature=0),
    tools=[chat_tool],
)

chat = model.start_chat()
Send an image of a fox, along with a simple prompt:
response = chat.send_message(
    [
        Part.from_uri(
            "gs://github-repo/generative-ai/gemini/function-calling/baby-fox.jpg",
            mime_type="image/jpeg",
        ),
        "Tell me about this animal",
    ]
)

response.candidates[0].function_calls
Now ask about the location details in the image:
response = chat.send_message(
    [
        Part.from_uri(
            "gs://github-repo/generative-ai/gemini/function-calling/baby-fox.jpg",
            mime_type="image/jpeg",
        ),
        "Tell me details about this location",
    ]
)

response.candidates[0].function_calls
And finally, ask for a color palette based the image:
response = chat.send_message(
    [
        Part.from_uri(
            "gs://github-repo/generative-ai/gemini/function-calling/baby-fox.jpg",
            mime_type="image/jpeg",
        ),
        "Get the color palette of this image and check it for accessibility",
    ]
)

response.candidates[0].function_calls
While this chatbot doesn't actually execute the predicted functions, it demonstrates creating an interactive experience using multimodal inputs and function calling in a chat format. You can extend this example by implementing REST API calls or client library requests for each function to create a truly functional and engaging multimodal chatbot that's connected to the real world.
## Conclusions

In this notebook, you explored the powerful capabilities of Gemini's multimodal function calling. You learned how to:

- Define functions and package them into tools.
- Send multimodal inputs (images, video, audio, PDFs) and prompts to Gemini. 
- Extract predicted function calls and their parameters.
- Use the predicted output to make (or potentially make) API calls.
- Return API responses to Gemini for natural language generation. 

You've seen how Gemini can understand and act on a range of different multimodal inputs, which opens up a world of possibilities for building innovative and engaging multimodal applications.  You can now use these powerful tools to create your own intelligent applications that seamlessly integrate media, natural language, and calls to external APIs and system.

Experiment with different modalities, functions, and prompts to discover the full potential of Gemini's multimodal and function calling capabilities. And you can continue learning by exploring other sample notebooks in this repository and exploring the [documentation for Gemini Function Calling](https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/function-calling). 