from flask import Flask, request, jsonify
from functools import wraps
from datetime import datetime
import uuid
import logging
from google.cloud.logging.handlers import CloudLoggingHandler
from google.cloud import logging as cloud_logging
from src.surfire2 import generate_mlb_analysis
from src.pod import generate_mlb_podcast_with_audio
from src.pall import generate_spanish_audio
from src.jap import generate_japanese_audio
from src.anchor import anchor, game_info_cache, get_last_x_games
import re
import os

app = Flask(__name__)

app_version = "0.0.0"

# Google Cloud Logging setup
client = cloud_logging.Client()
handler = CloudLoggingHandler(client, name="flask_app")
logger = logging.getLogger('flask_app')
logger.setLevel(logging.INFO)
logger.addHandler(handler)

# Add formatting for structured logging
formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)


#GET /api/v1/teams
#GET /api/v1/teams/<team_name>/players
#POST /api/v1/podcast
#GET /api/v1/teams/<team_name>/games/last
#GET /api/v1/teams/<team_name>/games/history?count=5

def validate_request(required_fields):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if request.is_json and request.method == 'POST':
                data = request.get_json()
                missing_fields = [field for field in required_fields if field not in data]
                if missing_fields:
                    return jsonify({
                        "error": "Missing required fields",
                        "missing_fields": missing_fields
                    }), 400
            return f(*args, **kwargs)
        return decorated_function
    return decorator

def error_handler(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {f.__name__}: {str(e)}")
            return jsonify({
                "error": "Internal server error",
                "message": str(e)
            }), 500
    return decorated_function


def get_players_for_team(team):
    """Fetch active players for a selected team using the analysis engine."""
    try:
        # Get response from analysis engine
        analysis_result = generate_mlb_analysis(f"List all current players on the {team} roster.")
        
        def clean_player_name(name):
            """Clean and standardize player names"""
            name = name.strip()
            name = re.sub(r'^[-*•⦁→▪️►]+\s*', '', name)  # Remove various bullet points
            name = re.sub(r'^\d+\.\s*', '', name)  # Remove numbering
            name = name.rstrip('.')  # Remove trailing periods
            name = re.sub(r'\s+', ' ', name)  # Standardize spacing
            name = re.sub(r'\([^)]*\)', '', name).strip()  # Remove parentheticals
            return name

        def split_player_names(text):
            """Split a string of space-separated names into individual player names"""
            # Split the text into potential name components
            words = text.split()
            players = []
            current_name = []
            
            for word in words:
                # Skip ellipsis and common separators
                if word in ['...', '…', '-', '...']:
                    if current_name:
                        players.append(' '.join(current_name))
                        current_name = []
                    continue
                
                current_name.append(word)
                
                # If we have two or more words, check if it's a complete name
                if len(current_name) >= 2:
                    # Check if the next word starts with a capital letter (likely a new name)
                    if (len(words) > len(players) + len(current_name) and 
                        words[len(players) + len(current_name)][0].isupper()):
                        players.append(' '.join(current_name))
                        current_name = []
            
            # Add the last name if any remains
            if current_name:
                players.append(' '.join(current_name))
            
            return players

        players = []
        
        # Strategy 1: Original bullet point parsing
        for line in analysis_result.split('\n'):
            if any(line.strip().startswith(bullet) for bullet in ['-', '*', '•', '⦁', '→', '▪️', '►']):
                player = clean_player_name(line)
                if player:
                    players.append(player)
        
        # Strategy 2: Try numbered list
        if not players:
            for line in analysis_result.split('\n'):
                if re.match(r'^\d+\.\s', line):
                    player = clean_player_name(line)
                    if player:
                        players.append(player)
        
        # Strategy 3: Comma separation
        if not players:
            text = analysis_result
            if ':' in text:
                text = text.split(':')[-1]
            text = re.sub(r'^.*?(includes|are|roster:)', '', text, flags=re.IGNORECASE)
            text = text.replace(' and ', ', ')
            players = [clean_player_name(p) for p in text.split(',') if clean_player_name(p)]
        
        # Strategy 4: Handle space-separated names
        if not players or len(players) == 1:
            # If we only found one "player" that's very long, it might be space-separated names
            text = players[0] if players else analysis_result
            if len(text.split()) > 4:  # If there are more than 4 words, likely multiple names
                players = split_player_names(text)
        
        # Validate and clean the final list
        cleaned_players = []
        seen_names = set()
        
        for player in players:
            clean_name = clean_player_name(player)
            # Validate name format
            if (clean_name and 
                len(clean_name.split()) >= 2 and  # Must be at least first and last name
                clean_name not in seen_names and
                len(clean_name) >= 4 and  # Arbitrary minimum length for valid name
                all(word[0].isupper() for word in clean_name.split())):  # Names should be capitalized
                
                cleaned_players.append(clean_name)
                seen_names.add(clean_name)
        
        if not cleaned_players:
            logging.warning(f"No valid players extracted for team {team}")
            return []
        
        # Sort alphabetically for consistent display
        return sorted(cleaned_players)

    except Exception as e:
        logging.error(f"Error in get_players_for_team for {team}: {str(e)}")
        return []

    except Exception as e:
        logging.error(f"Error in get_players_for_team for {team}: {str(e)}")
        return []


def get_mlb_teams():
    """Fetch all current MLB teams using the analysis engine."""
    try:
        analysis_result = generate_mlb_analysis("List all current MLB teams.")
        
        # Parse teams from bullet-point format
        teams = []
        for line in analysis_result.split('\n'):
            # Look for lines starting with dash/bullet
            if line.strip().startswith('-'):
                # Extract team name and clean it
                team = line.replace('-', '').strip()
                if team:  # Only add non-empty team names
                    teams.append(team)
        
        if not teams:  # If no teams found with bullet points, try alternate parsing
            # Try splitting by commas if it's a comma-separated list
            text = analysis_result.split(':')[-1] if ':' in analysis_result else analysis_result
            text = text.replace(' and ', ', ')
            teams = [team.strip() for team in text.split(',') if team.strip()]
        
        return teams
    except Exception as e:
        return []

def get_podcast_timestamp() -> dict:
    """
    Gets the current timestamp for podcast generation.
    
    Returns:
        dict: Current date and time information
    """
    current = datetime.now()
    return {
        "date": current.strftime("%Y-%m-%d"),
        "time": current.strftime("%H:%M:%S"),
        "timezone": current.astimezone().tzname(),
        "full_timestamp": current.strftime("%Y-%m-%d %H:%M:%S %Z")
    }


def construct_prompt(selected_team, selected_players, selected_timeframe, 
                 timeframe_value, selected_game_type, selected_opponent, 
                 selected_language):
# Get current timestamp
 timestamp = get_podcast_timestamp()

    # If last game is selected, get the exact date
 if selected_timeframe == "Last game":
        last_game_info = anchor(selected_team)
        if 'last_game_date' in last_game_info:
            timeframe_value = last_game_info['last_game_date']
    

 prompt_parts = [
                 f"Generate a podcast about the {selected_team}.",
                 f"Podcast generated on {timestamp['full_timestamp']}."
                ]

 # Players
 if selected_players:
     prompt_parts.append(f"Include highlights for players: {', '.join(selected_players)}.")

 # Timeframe
 if selected_timeframe == "Last game":
        prompt_parts.append(f"Cover the last game played by the {selected_team} on {timeframe_value}.")
 elif selected_timeframe == "Last X games":
     games_info = get_last_x_games(selected_team, timeframe_value)
     if 'games' in games_info:
            game_dates = [game['date'] for game in games_info['games']]     
            prompt_parts.append(f"Cover the last {timeframe_value} games played by the {selected_team} from {game_dates[-1]} to {game_dates[0]}.")
 elif selected_timeframe == "Specific date":
     prompt_parts.append(f"Cover the {selected_team} game on {timeframe_value}.")
 elif selected_timeframe == "Date Range":
     prompt_parts.append(
         f"Cover the {selected_team} games between {timeframe_value[0]} and {timeframe_value[1]}."
     )

 # Game Type
 if selected_game_type != "Any":
     prompt_parts.append(f"Focus on {selected_game_type.lower()} games.")

 # Opponent Team
 if selected_opponent != "Any":
     prompt_parts.append(f"Specifically include games against {selected_opponent}.")

 # Language
 prompt_parts.append(f"Generate the podcast script in {selected_language}.")

 return " ".join(prompt_parts)

@app.route('/api/v1/teams', methods=['GET'])
@error_handler
def get_teams():
    teams = get_mlb_teams()
    return jsonify({
        "status": "success",
        "data": {"teams": teams}
    })

@app.route('/api/v1/teams/<string:team_name>/players', methods=['GET'])
@error_handler
def get_team_players(team_name):
    players = get_players_for_team(team_name)
    return jsonify({
        "status": "success",
        "data": {
            "team": team_name,
            "players": players
        }
    })

@app.route('/api/v1/podcast', methods=['POST'])
@error_handler
@validate_request(['team', 'timeframe', 'game_type', 'language'])
def generate_podcast():
    data = request.get_json()
    
    def validate_timeframe(timeframe, value):
        if timeframe == "Last X games":
            if not isinstance(value, int) or value < 1:
                raise ValueError("timeframe_value must be a positive integer")
        elif timeframe == "Specific date":
            try:
                datetime.strptime(value, "%Y-%m-%d")
            except ValueError:
                raise ValueError("timeframe_value must be YYYY-MM-DD")
        elif timeframe == "Date Range":
            if not isinstance(value, dict) or 'start_date' not in value or 'end_date' not in value:
                raise ValueError("timeframe_value must include start_date and end_date")
            datetime.strptime(value['start_date'], "%Y-%m-%d")
            datetime.strptime(value['end_date'], "%Y-%m-%d")

    validate_timeframe(data['timeframe'], data.get('timeframe_value'))

    audio_generators = {
        "english": generate_mlb_podcast_with_audio,
        "japanese": generate_japanese_audio,
        "spanish": generate_spanish_audio
    }

    if data['language'].lower() not in audio_generators:
        return jsonify({
            "error": "Unsupported language",
            "supported_languages": list(audio_generators.keys())
        }), 400

    output_filename = f"podcast-{uuid.uuid4()}.mp3"
    contents = construct_prompt(
        data['team'],
        data.get('players', []),
        data['timeframe'],
        data.get('timeframe_value'),
        data['game_type'],
        data.get('opponent', "Any"),
        data['language']
    )

    audio_file = audio_generators[data['language'].lower()](
        contents,
        output_filename=output_filename
    )

    return jsonify({
        "status": "success",
        "data": {
            "audio_url": audio_file,
            "message": "Podcast generated successfully"
        }
    })

@app.route('/api/v1/teams/<string:team_name>/games/last', methods=['GET'])
@error_handler
def get_last_game(team_name):
    game_info = anchor(team_name)
    return jsonify({
        "status": "success",
        "data": game_info
    })

@app.route('/api/v1/teams/<string:team_name>/games/history', methods=['GET'])
@error_handler
def get_game_history(team_name):
    games_count = request.args.get('count', default=1, type=int)
    if games_count < 1:
        return jsonify({
            "error": "Invalid count parameter",
            "message": "Count must be a positive integer"
        }), 400

    games = get_last_x_games(team_name, games_count)
    return jsonify({
        "status": "success",
        "data": {
            "team": team_name,
            "games": games
        }
    })

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))