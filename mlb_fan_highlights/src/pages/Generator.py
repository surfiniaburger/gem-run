import streamlit as st
from surfire2 import generate_mlb_analysis
from pod import generate_mlb_podcast_with_audio
from pall import generate_spanish_audio
from jap import generate_japanese_audio
from firebase_config import get_auth, get_firestore
from datetime import datetime
from firebase_admin import firestore
from google.cloud import storage
import os
from google.api_core.exceptions import NotFound
import uuid
from datetime import timedelta
from google.cloud import logging as cloud_logging
import logging
from user_profile import UserProfile
import streamlit.components.v1 as components
import re
from anchor import anchor, game_info_cache, get_last_x_games

# Configure cloud logging at the top of the script, before other imports
logging.basicConfig(level=logging.INFO)
log_client = cloud_logging.Client()
log_client.setup_logging()

# Get Firebase services
auth = get_auth()
db = get_firestore()

# Inject GA script into Streamlit
st.set_page_config(
    page_title="MLB Podcast Generator",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants for Google Cloud Storage
GCS_BUCKET_NAME = "mlb-podcast-bucket" # Replace this with your desired bucket name, it should be unique
GCS_LOCATION = "US" # Replace this with desired location
GCS_PROJECT = "gem-rush-007" # Replace this with your desired project ID

def handle_authentication(email, password, auth_type):
 """Enhanced authentication handler with detailed error handling"""
 try:
     logging.info(f"Authentication attempt for {email} with type: {auth_type}")
     if auth_type == "Sign In":
         user = auth.get_user_by_email(email)
         auth_user = auth.get_user(user.uid)
         st.session_state['user'] = auth_user
         
         # Create/update user profile
         profile = UserProfile(user.uid, email)
         profile.create_or_update()
         
         st.success(f"Welcome back, {email}!")
         return True
         
     else:  # Sign Up
         # Password validation
         if len(password) < 6:
             st.error("Password must be at least 6 characters long")
             return False
             
         user = auth.create_user(email=email, password=password)
         auth_user = auth.get_user(user.uid)
         st.session_state['user'] = auth_user
         
         # Create new user profile
         profile = UserProfile(user.uid, email)
         profile.create_or_update({
             'account_type': 'free',
             'podcasts_generated': 0
         })
         
         st.success(f"Welcome to MLB Podcast Generator, {email}!")
         return True
         
 except auth.EmailAlreadyExistsError:
     st.error("This email is already registered. Please sign in instead.")
 except auth.UserNotFoundError:
     st.error("No account found with this email. Please sign up.")
 except auth.InvalidEmailError:
     st.error("Please enter a valid email address.")
 except auth.WeakPasswordError:
     st.error("Password is too weak. Please choose a stronger password.")
 except Exception as e:
     st.error(f"Authentication error: {str(e)}")
     logging.error(f"Authentication error for {email}: {str(e)}")
 return False


def sign_in_or_sign_up():
 """Enhanced sign in/sign up form with validation"""
 auth_type = st.radio("Sign In or Sign Up", ["Sign In", "Sign Up"])
 
 with st.form(key='auth_form'):
     email = st.text_input("Email")
     password = st.text_input("Password", type="password")
     submit_button = st.form_submit_button(auth_type)
     
     if submit_button:
         if not email or not password:
             st.error("Please fill in all fields.")
             return
         
         if handle_authentication(email, password, auth_type):
             # Use rerun() to refresh the page after successful authentication
             st.rerun()

def handle_logout():
 """Handles user logout"""
 if st.sidebar.button("Logout"):
     # Clear all session state
     for key in list(st.session_state.keys()):
         del st.session_state[key]
     # Use the current rerun() method instead of experimental_rerun()
     st.rerun()


@st.cache_data(ttl=3600)  # Cache for 1 hour
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
        st.error(f"Error fetching teams: {str(e)}")
        return []


@st.cache_data(ttl=3600)  # Cache for 1 hour
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
        st.error(f"Error fetching players: {str(e)}")
        logging.error(f"Error in get_players_for_team for {team}: {str(e)}")
        return []

    except Exception as e:
        st.error(f"Error fetching players: {str(e)}")
        logging.error(f"Error in get_players_for_team for {team}: {str(e)}")
        return []

def create_gcs_bucket(bucket_name, location):
 """Creates a Google Cloud Storage bucket if it doesn't exist."""
 storage_client = storage.Client(project = GCS_PROJECT)
 bucket = storage_client.bucket(bucket_name)
 
 try:
   storage_client.get_bucket(bucket_name)
   print(f"Bucket with name : {bucket_name} already exist")
 except NotFound:
   print(f"Creating bucket with name: {bucket_name}")
   bucket = storage_client.create_bucket(bucket, location=location)
   print(f"Bucket {bucket} created in {location}")
 except Exception as e:
   raise Exception(f"An error has occured while creating gcs bucket, : {e}")
 
def upload_audio_to_gcs(audio_content: bytes, file_name: str) -> str:
 """Uploads audio to GCS and returns a signed URL."""
 try:
   # Create a google cloud client
   storage_client = storage.Client(project = GCS_PROJECT)
   # Create or get the bucket
   create_gcs_bucket(GCS_BUCKET_NAME, GCS_LOCATION)
   bucket = storage_client.bucket(GCS_BUCKET_NAME)
   # Upload the file
   blob = bucket.blob(file_name)
   blob.upload_from_string(audio_content, content_type="audio/mp3")
   #Generate the signed URL
   url = blob.generate_signed_url(
     version="v4",
     expiration=timedelta(minutes=15),
     method="GET"
   )
   return url
 except Exception as e:
     raise Exception(f"An error occurred while uploading audio to GCS: {e}")


def main():
 # Inject the Google Analytics code
 components.html(
    """
    <!-- Google tag (gtag.js) -->
    <script async src="https://www.googletagmanager.com/gtag/js?id=G-98KGSC9LXG"></script>
    <script>
      window.dataLayer = window.dataLayer || [];
      function gtag(){dataLayer.push(arguments);}
      gtag('js', new Date());

      gtag('config', 'G-98KGSC9LXG');
    </script>
    """,
    height=0,
 )

 st.title("MLB Podcast Generator")
 st.write("Customize your MLB podcast by selecting your preferences below.")
 logging.info("MLB Podcast Generator application started")

 # Show logout button in sidebar if user is logged in
 if 'user' in st.session_state:
     handle_logout()
     
     # Get user profile
     profile = UserProfile(st.session_state['user'].uid, st.session_state['user'].email)
     user_limits = profile.check_usage_limits()
     user_data = profile.get_profile()

     if not user_limits['can_generate']:
        st.warning(f"You've reached your daily limit for your {user_limits['account_type']} account.")
        return
    
     if user_data:
         st.sidebar.write(f"Welcome, {user_data.get('email')}")
         if 'preferences' in user_data:
             st.sidebar.write("Your Preferences:")
             prefs = user_data['preferences']
             if 'favorite_team' in prefs:
                 st.sidebar.write(f"Favorite Team: {prefs['favorite_team']}")
             if 'preferred_language' in prefs:
                 st.sidebar.write(f"Preferred Language: {prefs['preferred_language']}")
            
             # Display usage statistics
             usage_stats = profile.get_usage_stats()
             if usage_stats:
                 st.sidebar.write("Usage Statistics:")
                 st.sidebar.write(f"Podcasts Generated: {usage_stats['podcasts_generated']}")
                 if usage_stats['account_created']:
                     st.sidebar.write(f"Account Created: {usage_stats['account_created'].strftime('%Y-%m-%d')}")
 

 # If a user does not exist in the session, create authentication
 if 'user' not in st.session_state:
   sign_in_or_sign_up()
   return
 else:
   st.write(f"Logged in as: {st.session_state['user'].email}")
 
 # Fetch MLB teams
 mlb_teams = get_mlb_teams()
 
 if not mlb_teams:
     st.error("Unable to fetch MLB teams. Please try again later.")
     return

 with st.expander("Customize your podcast options", expanded=True):
     # Primary team selection
     selected_team = st.selectbox("Select Primary Team", [""] + mlb_teams)
     print(f"Selected team: {selected_team}")

     # Player Selection (dependent on team selection)
     selected_players = []
     if selected_team:
         available_players = get_players_for_team(selected_team)
         if available_players:
             selected_players = st.multiselect(
                 "Select players to highlight (optional)",
                 available_players
             )
         else:
             st.warning(f"Unable to fetch players for {selected_team}.")

     # Timeframe Selection
     timeframe_options = ["Last game", "Last X games", "Specific date", "Date Range"]
     selected_timeframe = st.selectbox("Select Timeframe", timeframe_options)

     timeframe_value = None
     if selected_timeframe == "Last X games":
         timeframe_value = st.number_input("Enter number of games", min_value=1, max_value=162, step=1)
     elif selected_timeframe == "Specific date":
         timeframe_value = st.date_input("Select date")
     elif selected_timeframe == "Date Range":
         col1, col2 = st.columns(2)
         with col1:
             start_date = st.date_input("Start Date")
         with col2:
             end_date = st.date_input("End Date")
         timeframe_value = (start_date, end_date)

     # Game Type Selection
     game_type_options = ["Any", "Regular season", "World Series", "Spring Training"]
     selected_game_type = st.selectbox("Select Game Type", game_type_options)

     # Opponent Team Selection
     opponent_teams = [team for team in mlb_teams if team != selected_team]
     selected_opponent = st.selectbox("Select Opponent Team (optional)", ["Any"] + opponent_teams)

     # Language selection
     language_options = ["English", "Spanish", "Japanese"]
     selected_language = st.selectbox("Select preferred language", language_options)

 if selected_team or selected_language:
     profile = UserProfile(st.session_state['user'].uid, 
                         st.session_state['user'].email)
     profile.update_preferences(selected_team, selected_language)

 # Generate Podcast Button
 if not selected_team:
     st.warning("Please select a team to generate a podcast.")
 elif st.button("Generate Podcast"):
     with st.spinner("Generating podcast..."):
         contents = construct_prompt(
             selected_team,
             selected_players,
             selected_timeframe,
             timeframe_value,
             selected_game_type,
             selected_opponent,
             selected_language
         )
         
         try:
             # Select the appropriate audio generation function based on language
             audio_file = None
             profile = UserProfile(st.session_state['user'].uid, 
                         st.session_state['user'].email)
             if selected_language.lower() == "english":
                 audio_file = generate_mlb_podcast_with_audio(
                     contents,
                     output_filename=f"podcast-{uuid.uuid4()}.mp3"
                 )
             elif selected_language.lower() == "japanese":
                  audio_file = generate_japanese_audio(
                     contents,
                     language=selected_language,
                      output_filename=f"podcast-{uuid.uuid4()}.mp3"
                 )
             elif selected_language.lower() == "spanish":
                  audio_file = generate_spanish_audio(
                     contents,
                     language=selected_language,
                     output_filename=f"podcast-{uuid.uuid4()}.mp3"
                 )
             else:
               raise ValueError(f"Unsupported language: {selected_language}")
             
             # Store the audio url
             if audio_file:
                profile.store_podcast(audio_file)
                profile.increment_podcasts_generated()
                st.audio(audio_file)
             
         except Exception as e:
             st.error(f"An error occurred while generating {selected_language} audio: {str(e)}")


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

if __name__ == "__main__":
 main()