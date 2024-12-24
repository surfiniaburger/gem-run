# streamlit_app.py
import streamlit as st
from mlb_core import MLBHighlightGenerator

def main():
    st.title("MLB Game Highlights Generator")
    
    # Initialize the core functionality
    mlb_generator = MLBHighlightGenerator()
    
    # Get teams data
    teams_data = mlb_generator.get_teams_data()
    
    if teams_data is None:
        st.error("Failed to retrieve team data.")
        return
    
    # Create team selection dropdown
    team_names = sorted(teams_data['name'].tolist())
    selected_team = st.selectbox("Select an MLB Team", team_names)
    
    if st.button("Generate Highlight"):
        with st.spinner("Generating highlight..."):
            # Get team ID
            selected_team_id = teams_data[teams_data['name'] == selected_team]['id'].iloc[0]
            
            # Get team games
            team_games = mlb_generator.get_team_games(selected_team_id)
            
            if not team_games:
                st.warning(f"No games found for {selected_team}")
                return
            
            # Get most recent game
            recent_game = team_games[-1]
            
            # Generate highlight
            highlight = mlb_generator.generate_highlight(recent_game)
            
            # Display result
            st.subheader("Game Highlight")
            st.write(highlight)

if __name__ == "__main__":
    main()
