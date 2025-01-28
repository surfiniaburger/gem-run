import streamlit as st
import asyncio
from embed import PlayerImageSimilarity, main

# Set up the Streamlit app
st.title("MLB Player Similarity Search")
st.write("Enter an MLB player's name or click the button to find similar players.")

# Input field for the player's name
player_name = st.text_input("Enter MLB Player Name (e.g., A.J._Minter_ATL_2024.jpg):")

# Button to trigger the similarity search
if st.button("Find Similar Players"):
    if player_name:
        st.write(f"Searching for players similar to: {player_name}")

        # Initialize the PlayerImageSimilarity class
        similarity_search = PlayerImageSimilarity(
            project_id="gem-rush-007",
            bucket_name="mlb-headshots",
            region="us-central1",
            collection_name="player_embeddings",
            secret_id="cloud-run-invoker"
        )

        # Verify bucket access
        similarity_search.verify_bucket_access()

        # Perform the similarity search
        try:
            similar_players = asyncio.run(similarity_search.find_similar_players(player_name))
            
            # Display the results
            if similar_players:
                st.write("### Similar Players Found:")
                for player in similar_players:
                    st.write(f"**Player Name:** {player['player_name']}")
                    st.write(f"**Team:** {player['team']}")
                    st.write(f"**Year:** {player['year']}")
                    st.write(f"**File Name:** {player['file_name']}")
                    st.write(f"**GCS URI:** {player['gcs_uri']}")
                    st.write("---")
            else:
                st.write("No similar players found.")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("Please enter a valid MLB player name.")