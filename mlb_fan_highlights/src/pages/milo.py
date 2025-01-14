import streamlit as st
from surfire2 import generate_mlb_analysis
from pod import generate_mlb_podcast_with_audio

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
                audio_file = generate_mlb_podcast_with_audio(
                contents,
                language=selected_language,  # Pass the selected language
                output_filename="mlb_podcast.mp3"
            )
                st.audio(audio_file)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

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