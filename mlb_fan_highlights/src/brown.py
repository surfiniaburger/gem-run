import streamlit as st
import asyncio
import logging
from embed import PlayerImageSimilarity

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up the Streamlit app
st.title("MLB Player Similarity Search")
st.write("Enter an MLB player's name or click the button to find similar players.")

# Add a debug section that can be toggled
show_debug = st.sidebar.checkbox("Show Debug Information")

# Input field for the player's name
player_name = st.text_input("Enter MLB Player Name (e.g., A.J._Minter_ATL_2024.jpg):")

# Button to trigger the similarity search
if st.button("Find Similar Players"):
    if player_name:
        st.write(f"Searching for players similar to: {player_name}")
        logger.info(f"Starting similarity search for player: {player_name}")
        
        try:
            # Initialize the PlayerImageSimilarity class
            similarity_search = PlayerImageSimilarity(
                project_id="gem-rush-007",
                bucket_name="mlb-headshots",
                region="us-central1",
                collection_name="player_embeddings",
                secret_id="cloud-run-invoker"
            )
            logger.info("PlayerImageSimilarity initialized successfully")
            
            # Verify bucket access
            similarity_search.verify_bucket_access()
            logger.info("Bucket access verified")
            
            # Perform the similarity search
            similar_players = asyncio.run(similarity_search.find_similar_players(player_name))
            logger.info(f"Search completed. Found {len(similar_players)} results")
            
            # Display debug information if enabled
            if show_debug:
                st.write("### Debug Information")
                st.write("Raw results object type:", type(similar_players))
                st.write("Number of results:", len(similar_players))
                
                for i, player in enumerate(similar_players):
                    st.write(f"\nResult {i+1} Debug Info:")
                    st.write("Result type:", type(player))
                    st.write("Available attributes:", dir(player))
                    if hasattr(player, 'metadata'):
                        st.write("Metadata:", player.metadata)
                    if hasattr(player, 'page_content'):
                        st.write("Page content:", player.page_content)
                    st.write("---")
                
                logger.info("Debug information displayed")
            
            # Display the results
            if similar_players:
                st.write("### Similar Players Found:")
                for i, player in enumerate(similar_players, 1):
                    logger.info(f"Processing result {i}")
                    
                    # Extract metadata from the Document object
                    metadata = player.metadata if hasattr(player, 'metadata') else {}
                    logger.debug(f"Extracted metadata: {metadata}")
                    
                    # Create an expander for each player
                    with st.expander(f"Player {i}"):
                        # Display player information
                        st.write("#### Player Details")
                        if 'player_name' in metadata:
                            st.write(f"**Player Name:** {metadata['player_name']}")
                        if 'team' in metadata:
                            st.write(f"**Team:** {metadata['team']}")
                        if 'year' in metadata:
                            st.write(f"**Year:** {metadata['year']}")
                        if 'file_name' in metadata:
                            st.write(f"**File Name:** {metadata['file_name']}")
                        if 'gcs_uri' in metadata:
                            st.write(f"**GCS URI:** {metadata['gcs_uri']}")
                        if hasattr(player, 'page_content'):
                            st.write("**Embedding Content:**", player.page_content[:100] + "..." if len(player.page_content) > 100 else player.page_content)
                        
                        # Display any additional metadata fields
                        if show_debug:
                            st.write("#### Additional Metadata")
                            for key, value in metadata.items():
                                if key not in ['player_name', 'team', 'year', 'file_name', 'gcs_uri']:
                                    st.write(f"**{key}:** {value}")
                    
                    logger.info(f"Displayed information for result {i}")
            else:
                st.write("No similar players found.")
                logger.warning("No similar players found in search results")
                
        except Exception as e:
            error_msg = f"An error occurred: {str(e)}"
            logger.error(error_msg, exc_info=True)
            st.error(error_msg)
            
            if show_debug:
                st.write("### Error Details")
                import traceback
                st.code(traceback.format_exc())
    else:
        st.warning("Please enter a valid MLB player name.")
        logger.warning("Search attempted with empty player name")

# Add logging configuration information
if show_debug:
    st.sidebar.write("### Logging Configuration")
    st.sidebar.write(f"Logger name: {logger.name}")
    st.sidebar.write(f"Logger level: {logging.getLevelName(logger.level)}")
    st.sidebar.write("Handler levels:")
    for handler in logger.handlers:
        st.sidebar.write(f"- {type(handler).__name__}: {logging.getLevelName(handler.level)}")