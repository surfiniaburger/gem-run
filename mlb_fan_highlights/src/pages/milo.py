import streamlit as st
from surfire import generate_mlb_analysis  # Assuming your generate_mlb_analysis is in surfire.py
import json

def main():
    st.title("MLB Podcast Generator")
    st.write("Customize your MLB podcast by selecting your preferences below.")

    # --- User Input ---
    with st.expander("Customize your podcast options"):
          # Player Selection
          player_options = [
              "Shohei Ohtani",
              "Mookie Betts",
              "Freddie Freeman",
              "Max Muncy",
              "Will Smith",
              "Gavin Lux",
          ]  # Replace with a dynamic list if needed
          selected_players = st.multiselect("Select players (optional)", player_options,default=[])

          # Timeframe Selection
          timeframe_options = ["Last game", "Last X games", "Specific date", "Date Range"]
          selected_timeframe = st.selectbox("Select Timeframe", timeframe_options)

          timeframe_value = None
          if selected_timeframe == "Last X games":
               timeframe_value = st.number_input("Enter number of games", min_value=1, step=1)
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
          game_type_options = [
              "Any",
              "Regular season",
              "World Series",
               "Spring Training",
          ]  # Add additional game types
          selected_game_type = st.selectbox("Select Game Type", game_type_options)

          # Team Selection
          team_options = [
               "Any",
               "New York Yankees",
               "Boston Red Sox",
               "San Francisco Giants",
              "Los Angeles Angels",
               "New York Mets",

          ]
          selected_team = st.selectbox("Select Opponent Team (optional)", team_options)

          #Language selection
          language_options = ["English", "Spanish", "Japanese"]
          selected_language = st.selectbox("Select preferred language", language_options)

    # --- Generate Podcast Button ---
    if st.button("Generate Podcast Script"):
        with st.spinner("Generating podcast script..."):
            contents = construct_prompt(
                selected_players,
                selected_timeframe,
                timeframe_value,
                selected_game_type,
                selected_team,
                selected_language
            )
            try:
                result = generate_mlb_analysis(contents)
                if isinstance(result, dict) and "error" in result:
                  st.error(f"Error: {result['error']}")
                else:
                   st.success("Podcast script generated successfully!")
                   st.json(result)

            except Exception as e:
                 st.error(f"An error occurred: {e}")

def construct_prompt(selected_players, selected_timeframe, timeframe_value, selected_game_type, selected_team,selected_language):
    """Constructs the prompt for the podcast agent based on user inputs."""
    prompt_parts = []

    # Team and Player
    if selected_players:
        prompt_parts.append(f"Include highlights for players: {', '.join(selected_players)}.")

    # Timeframe
    if selected_timeframe == "Last game":
        prompt_parts.append("Generate a podcast for the last game played by the Dodgers.")
    elif selected_timeframe == "Last X games":
         prompt_parts.append(f"Generate a podcast for the last {timeframe_value} games played by the Dodgers.")
    elif selected_timeframe == "Specific date":
         prompt_parts.append(f"Generate a podcast for the Dodgers game on {timeframe_value}.")
    elif selected_timeframe == "Date Range":
         prompt_parts.append(f"Generate a podcast for the Dodgers games between {timeframe_value[0]} and {timeframe_value[1]}.")


    # Game Type
    if selected_game_type != "Any":
        prompt_parts.append(f"Focus on {selected_game_type.lower()} games.")

    # Opponent Team
    if selected_team != "Any":
        prompt_parts.append(f"Specifically include games against {selected_team}.")

    #Language support
    prompt_parts.append(f"Translate the podcast script to {selected_language}.")

    full_prompt = " ".join(prompt_parts)
    return full_prompt

if __name__ == "__main__":
    main()