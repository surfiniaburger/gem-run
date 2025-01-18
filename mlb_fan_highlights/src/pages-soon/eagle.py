import streamlit as st
from surfire import generate_mlb_analysis 


@st.cache_data(ttl=3600)  # Cache for 1 hour (3600 seconds)
# Simulated function to get players for a selected team
def get_players_for_team(team):
    # This would be replaced with an actual API call or database query
    prompt = f"List all players on the {team} roster."
    analysis_result = generate_mlb_analysis(prompt)
    # Assuming the result is a string with player names separated by commas
    # Clean up the result
    players_list = analysis_result.split(':')[-1].strip()  # Remove the introductory text
    players = [player.strip() for player in players_list.split(',')]
    
    return players

# List of MLB teams (you would need to keep this updated)
mlb_teams = ["Dodgers", "Yankees", "Red Sox", "Cubs", "Mets"]  # Add all MLB teams

# Streamlit app
st.title("MLB Data Retrieval")

# Team selection
selected_team = st.selectbox("Select a team:", [""] + mlb_teams)

# Player selection (dependent on team selection)
if selected_team:
    players = get_players_for_team(selected_team)
    selected_players = st.multiselect("Select players:", players)
else:
    st.warning("Please select a team before choosing players.")

# Timeframe selection
timeframe_option = st.radio("Select timeframe:", 
                            ["Last game", "Last X games", "Specific date", "Date Range"])

if timeframe_option == "Last X games":
    num_games = st.number_input("Number of games:", min_value=1, max_value=162)
elif timeframe_option == "Specific date":
    specific_date = st.date_input("Select date:")
elif timeframe_option == "Date Range":
    start_date = st.date_input("Start date:")
    end_date = st.date_input("End date:")

# Game Type selection
game_types = ["Regular season", "World Series", "Spring Training"]
game_type = st.selectbox("Game Type (optional):", [""] + game_types)

# Matchup selection
matchup_team = st.selectbox("Matchup (optional):", [""] + [team for team in mlb_teams if team != selected_team])

# Submit button
if st.button("Retrieve Data"):
    # Here you would process the inputs and call your data retrieval function
    st.success("Data retrieval initiated with the following parameters:")
    st.write(f"Team: {selected_team}")
    st.write(f"Players: {', '.join(selected_players) if selected_players else 'All'}")
    st.write(f"Timeframe: {timeframe_option}")
    if timeframe_option == "Last X games":
        st.write(f"Number of games: {num_games}")
    elif timeframe_option == "Specific date":
        st.write(f"Date: {specific_date}")
    elif timeframe_option == "Date Range":
        st.write(f"Date range: {start_date} to {end_date}")
    st.write(f"Game Type: {game_type if game_type else 'All'}")
    st.write(f"Matchup: {matchup_team if matchup_team else 'Any'}")