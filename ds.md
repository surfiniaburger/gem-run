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