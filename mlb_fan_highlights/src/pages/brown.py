import streamlit as st
import asyncio
from alloydb import create_player_embeddings_workflow

async def run_workflow():
    workflow = await create_player_embeddings_workflow()
    return workflow

def app():
    st.title("Player Embeddings Generator")
    
    if st.button("Generate Embeddings"):
        st.info("Processing... This may take a few minutes.")
        
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