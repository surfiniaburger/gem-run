import streamlit as st
from surfire2 import generate_mlb_analysis
from pod import generate_mlb_podcast_with_audio
from pall import generate_spanish_audio
from jap import generate_japanese_audio
from firebase_config import firebase_admin
from firebase_admin import auth

def create_user_profile(uid, email):
    """Creates a user profile in firebase, if one does not exist"""
    try:
        # Get the user, if user does not exist, this will throw exception
        user = auth.get_user(uid)
        print(f"user found, user profile : {user}")
    except auth.UserNotFoundError:
        # If not create the user profile, with email as additional information
        user = auth.create_user(uid=uid, email = email)
        print(f"user created, user profile : {user}")
    except Exception as e:
        print(f"error while creating user profile {e}")

def sign_in_or_sign_up():
    """Displays sign in or sign up form."""
    
    auth_type = st.radio("Sign In or Sign Up", ["Sign In", "Sign Up"])
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if st.button(auth_type):
            try:
                if auth_type == "Sign In":
                  user = auth.get_user_by_email(email)
                  auth_user = auth.get_user(user.uid)
                  # Handle sign in logic (not directly within Streamlit, this is done in firebase, we just need to verify user)
                  st.session_state['user'] = auth_user
                  st.success(f"Signed in successfully as {email}")
                  create_user_profile(user.uid, email)
                else:
                  user = auth.create_user(email=email, password=password)
                  auth_user = auth.get_user(user.uid)
                  # Handle sign up logic (not directly within Streamlit, this is done in firebase, we just need to verify user)
                  st.session_state['user'] = auth_user
                  st.success(f"Signed up successfully as {email}")
                  create_user_profile(user.uid, email)

            except auth.EmailAlreadyExistsError:
                 st.error(f"Email already exist. Please sign in")
            except auth.UserNotFoundError:
                 st.error(f"No user with this email found. Please sign up or check your email")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_mlb_teams():
    """Fetch all current MLB teams using the analysis engine."""
    prompt = "List all current MLB teams."
    try:
        analysis_result = generate_mlb_analysis(prompt)
        # Assuming the result is a string with team names separated by commas
        teams_list = analysis_result.split(':')[-1].strip()
        # Handle the "and" in the list by replacing it with a comma
        teams_list = teams_list.replace(' and ', ', ')
        # Split and clean the list
        return [team.strip() for team in teams_list.split(',') if team.strip()]
    except Exception as e:
        st.error(f"Error fetching teams: {str(e)}")
        return []

@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_players_for_team(team):
    """Fetch active players for a selected team using the analysis engine."""
    prompt = f"List all current players on the {team} roster."
    try:
        analysis_result = generate_mlb_analysis(prompt)
        # Assuming the result is a string with player names separated by commas
        players_list = analysis_result.split(':')[-1].strip()
        # Handle the "and" in the list by replacing it with a comma
        players_list = players_list.replace(' and ', ', ')
        # Split and clean the list
        return [player.strip() for player in players_list.split(',') if player.strip()]
    except Exception as e:
        st.error(f"Error fetching players: {str(e)}")
        return []

def main():
    st.title("MLB Podcast Generator")
    st.write("Customize your MLB podcast by selecting your preferences below.")

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
                if selected_language.lower() == "english":
                    audio_file = generate_mlb_podcast_with_audio(
                        contents,
                        output_filename="mlb_podcast.mp3"
                    )
                elif selected_language.lower() == "japanese":
                     audio_file = generate_japanese_audio(
                        contents,
                        language=selected_language,
                        output_filename="mlb_podcast.mp3"
                    )
                elif selected_language.lower() == "spanish":
                     audio_file = generate_spanish_audio(
                        contents,
                        language=selected_language,
                        output_filename="mlb_podcast.mp3"
                    )
                else:
                  raise ValueError(f"Unsupported language: {selected_language}")
                
                st.audio(audio_file)
            
            except Exception as e:
                st.error(f"An error occurred while generating {selected_language} audio: {str(e)}")

def construct_prompt(selected_team, selected_players, selected_timeframe, 
                    timeframe_value, selected_game_type, selected_opponent, 
                    selected_language):
    """Constructs the prompt for the podcast agent based on user inputs."""
    prompt_parts = [f"Generate a podcast about the {selected_team}."]

    # Players
    if selected_players:
        prompt_parts.append(f"Include highlights for players: {', '.join(selected_players)}.")

    # Timeframe
    if selected_timeframe == "Last game":
        prompt_parts.append(f"Cover the last game played by the {selected_team}.")
    elif selected_timeframe == "Last X games":
        prompt_parts.append(f"Cover the last {timeframe_value} games played by the {selected_team}.")
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