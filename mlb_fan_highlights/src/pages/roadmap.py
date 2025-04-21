import streamlit as st

st.set_page_config(page_title="Project Roadmap", layout="wide")

st.title("🗺️ AI Highlights Agent: Project Roadmap")

st.markdown("""
This page outlines the current capabilities and future direction of the MLB AI Highlights Generator project.
Our goal is to create a fully automated system that transforms raw MLB game data into engaging, multi-asset media summaries.
""")

st.markdown("---")

# --- Current Status ---
st.header("✅ Current Status (As of April 2025 - Demo)")

st.markdown("""
The agent currently performs the following steps upon receiving a target game (identified by team selection):

1.  **Planning:** Analyzes the request and plans data retrieval steps.
2.  **Data Retrieval:**
    *   Fetches structured game metadata and play-by-play data from BigQuery.
    *   Performs vector search on BigQuery for relevant narrative snippets (summaries, etc.).
    *   Looks up player IDs and names from a metadata table.
3.  **Content Generation (Dialogue):** Generates an initial two-host dialogue script based on retrieved data.
4.  **Web Search & Refinement:**
    *   Critiques the initial draft for accuracy, engagement, and data usage.
    *   If needed, generates web search queries (using Tavily) to fetch external context based on the critique.
    *   Revises the dialogue script, incorporating critique and web search results. (Loops up to 2 times).
5.  **Static Asset Retrieval:**
    *   Analyzes the final script for team and player names.
    *   Retrieves team logos via vector search (against an indexed logo dataset).
    *   Retrieves player headshots via direct lookup (using Player ID and GCS path).
6.  **Generative Visuals:**
    *   Analyzes the final script to create descriptive text prompts for key moments (avoiding names/teams for model safety).
    *   Generates images using **Vertex AI Imagen 3**. Includes fallback to Cloudflare Workers AI if Imagen fails.
    *   Critiques the generated images against the script. (Loops up to 2 times).
    *   Generates short video clips (Text-to-Video) using **Vertex AI Veo** based on the image prompts.
7.  **Audio Generation:**
    *   Generates a multi-speaker audio track (**Vertex AI TTS**) matching the final dialogue script, alternating voices line-by-line.
8.  **Output:** Collects and presents the final script, retrieved static assets (logos, headshots), generated images, generated video clips, and the generated audio file URI.

**Data Source:** Primarily relies on 2024 MLB season data loaded into BigQuery tables for this demonstration phase.
""")

st.markdown("---")

# --- Roadmap Visualization ---
st.header("🛣️ Development Roadmap")

# Using Mermaid syntax for the flowchart within graphviz_chart
# (Streamlit renders Mermaid within this component)
roadmap_diagram = """
graph TD
    A[User Input: Team Selection] --> B(Agent Core);

    subgraph Agent Core
        direction LR
        B1[1. Planning] --> B2(2. Data Retrieval: BQ + Vector Search);
        B2 --> B3(3. Dialogue Script Generation);
        B3 --> B4{4. Refinement Loop};
        B4 -- Critique Needed? --> B5(Web Search);
        B5 --> B6(Revise Script);
        B6 --> B4;
        B4 -- Script OK --> C(5. Static Asset Retrieval);
    end

    C --> D(6. Generative Visuals);
    subgraph Generative Visuals
        direction LR
        D1[Prompt Analysis] --> D2(Imagen 3 / Cloudflare);
        D2 --> D3{Critique Loop};
        D3 -- Needs Improvement --> D2;
        D3 -- Visuals OK --> D4(Veo Text-to-Video);
    end

    D --> E(7. Multi-Speaker Audio Gen);
    E --> F(8. Asset Aggregation);

    subgraph Current Output [✅ Current Stage]
        F --> G(Display Assets: Script, Audio URI, Image URIs, Video URIs);
    end

    subgraph Next Steps [➡️ Next Steps / Future]
        direction TB
        F --> H(Assembly Line);
        H --> I{Output Format};
        I -- Video --> J[Standard Video Render (moviepy/ffmpeg)];
        I -- Interactive --> K[Interactive HTML/JS Card];
    end

    subgraph Long-Term Vision [🚀 Future Enhancements]
        J --> L[Advanced Editing / Effects];
        K --> M[Deeper Stat Integration / Viz];
        H --> N[Alternative Output Styles (e.g., Article)];
        B2 --> O[Real-time Data Feeds];
    end

    style Current Output fill:#d2f7d2,stroke:#333,stroke-width:2px
    style Next Steps fill:#d2e7f7,stroke:#333,stroke-width:2px
    style Long-Term Vision fill:#f7f7d2,stroke:#333,stroke-width:2px

"""

st.graphviz_chart(roadmap_diagram, use_container_width=True)


st.markdown("---")

# --- Detailed Future Plans ---
st.header("➡️ Next Steps & Future Plans")

st.subheader("Immediate Focus: Assembly Line")
st.markdown("""
*   **Goal:** Automatically combine the generated assets (script, audio, images, videos) into a cohesive final product.
*   **Implementation:** Develop a final agent node or separate process that acts as an "assembly line".
    *   **Challenge 1: Synchronization:** Accurately timing visual changes (showing specific images/videos) to match the corresponding points in the audio narration. This might involve:
        *   Timestamping the script using the TTS API's capabilities (if available).
        *   Using another LLM call to analyze the script and audio to create a detailed storyboard/edit decision list (EDL).
    *   **Challenge 2: Tooling:** Utilizing libraries like `moviepy` (Python) or orchestrating `ffmpeg` commands to perform the video editing tasks (overlaying images, inserting video clips, adding audio).
*   **Potential Outputs:**
    *   **Standard Video File (.mp4):** A straightforward video recap suitable for sharing. Images might use simple effects (like Ken Burns pan/zoom).
    *   **Interactive HTML/JS Card:** A web-based component where the script text is displayed, audio plays, and the relevant visual asset appears alongside the text dynamically. Offers more user engagement.

""")

st.subheader("Long-Term Vision")
st.markdown("""
*   **Real-time Capability:** Integrate with live MLB data APIs for near real-time recap generation.
*   **Deeper Analysis & Storytelling:** Incorporate more advanced stats (wOBA, FIP, etc.), historical context, and automated detection of narrative arcs within the game.
*   **Advanced Visuals:** Explore generating visualizations of player tracking data (e.g., pitch paths, defensive routes) or using more sophisticated video editing techniques.
*   **Customization & Control:** Allow users more control over the desired style, length, and focus of the generated highlights.
*   **Alternative Output Formats:** Generate written articles, social media snippets, or even VR experiences based on the game data and generated assets.
""")

st.markdown("---")
st.caption("This roadmap is subject to change based on development progress and evolving AI capabilities.")