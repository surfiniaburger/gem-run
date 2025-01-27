import streamlit as st
import asyncio
from alloydb import create_player_embeddings_workflow
import os
import logging

def setup_logger(log_level: str = "INFO") -> logging.Logger:
    """Set up a logger with the specified log level and formatting."""
    # Create logger
    logger = logging.getLogger("PlayerEmbeddings")
    logger.setLevel(getattr(logging, log_level))

    # Create console handler with formatting
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)

    # Add handler to logger if it doesn't already have handlers
    if not logger.handlers:
        logger.addHandler(console_handler)

    return logger

async def run_workflow():
    workflow = await create_player_embeddings_workflow(config_path="/app/mlb_fan_highlights/src/config.yaml")
    return workflow

def app():
    st.title("Player Embeddings Generator")
    
    if st.button("Generate Embeddings"):
        st.info("Processing... This may take a few minutes.")
        logger = setup_logger()
        logger.info("Starting the main app")
        os.chdir("/app/mlb_fan_highlights/src") # added to change the directory to find config.yaml
        
        # Run async workflow
        result = asyncio.run(run_workflow())
        
        if result:
            st.success("Embeddings generated successfully!")
            
            # Search functionality
            st.subheader("Search Similar Players")
            player_file = st.text_input("Enter player filename (e.g., A.J._Minter_ATL_2024.jpg)")
            k_similar = st.slider("Number of similar players", 1, 10, 5)
            
            if st.button("Search"):
                similar_players = asyncio.run(result["find_similar"](player_file, k_similar))
                
                st.write("Similar Players:")
                for player in similar_players:
                    st.write(player)

if __name__ == "__main__":
    app()