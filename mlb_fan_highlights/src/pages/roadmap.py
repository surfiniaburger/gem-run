# roadmap.py
import streamlit as st

st.set_page_config(page_title="Project Roadmap", layout="wide")

st.title("🗺️ AI Highlights Agent: Project Roadmap")

st.markdown("""
This page outlines the current capabilities and future direction of the MLB AI Highlights Generator project.
Our goal is to create a fully automated system that transforms raw MLB game data into engaging, multi-asset media summaries, culminating in a final video product.
""")

st.markdown("---")

# --- Current Status ---
st.header("✅ Current Status (As of May 2025 - Video Output Achieved)")

st.markdown("""
The agent currently performs the following steps upon receiving a target game (identified by team selection):

1.  **Planning:** Analyzes the request and plans data retrieval and generation steps.
2.  **Data Retrieval:**
    *   Fetches structured game metadata and play-by-play data from BigQuery.
    *   Performs vector search on BigQuery for relevant narrative snippets (summaries, etc.).
    *   Looks up player IDs and names from a metadata table for headshot retrieval.
3.  **Content Generation & Refinement (Dialogue):**
    *   Generates an initial two-host dialogue script based on retrieved data.
    *   Critiques the initial draft for accuracy, engagement, and data usage.
    *   If needed, generates web search queries (using Tavily) to fetch external context based on the critique.
    *   Revises the dialogue script, incorporating critique and web search results. (Loops up to 2 times).
4.  **Static Asset Retrieval:**
    *   Analyzes the final script for team and player names.
    *   Retrieves team logos via vector search (against an indexed logo dataset).
    *   Retrieves player headshots via direct GCS path lookup (using Player ID).
5.  **Generative Visuals:**
    *   Analyzes the final script to create descriptive text prompts for key moments (avoiding names/teams for model safety).
    *   Generates images using **Vertex AI Imagen 3**. Includes fallback to Cloudflare Workers AI.
    *   Critiques the generated images against the script. (Loops up to 2 times, regenerating based on critique).
    *   Generates short video clips (Text-to-Video) using **Vertex AI Veo** based on the final image prompts.
6.  **Audio Generation & Processing:**
    *   Generates a multi-speaker audio track (**Vertex AI TTS**) matching the final dialogue script, alternating voices line-by-line.
    *   Transcribes the generated audio using **Vertex AI Speech-to-Text** to obtain word-level timestamps.
7.  **Asset Aggregation:** Consolidates the final script, all retrieved/generated images, generated video clips, audio URI, and word timestamps.
8.  **Video Assembly:**
    *   Uses an LLM (**Gemini Flash/Pro**) to create a visual timeline, mapping specific images/videos to dialogue segments based on word timestamps.
    *   Downloads all necessary assets (audio, images, video clips).
    *   Uses **MoviePy** to composite the visual assets onto the timeline, synchronized with the audio track.
    *   Exports the final assembled video to a .mp4 file.
9.  **Output:** Uploads the final video to Google Cloud Storage and provides a playable video in the UI via a signed URL. Intermediate assets (script, audio, images, clips) are available in an expandable section for review.

**Data Source:** Primarily relies on 2024 MLB season data loaded into BigQuery tables for this demonstration phase. Uses GCS for storing static assets, generated assets, and final videos.
""")

st.markdown("---")

# --- Roadmap Visualization ---
st.header("🛣️ Development Process Flow")

# Updated Mermaid diagram reflecting the current implemented pipeline
roadmap_diagram = """
graph TD
    A[User Input: Team Selection] --> B(Agent Core);

    subgraph Agent Core
        direction LR
        B1[1. Planning] --> B2(2. Data Retrieval);
        B2 --> B3(3. Dialogue Script Gen + Refinement);
        B3 -- Final Script --> C(4. Static Asset Retrieval);
        C --> D(5. Generative Visuals Loop);
        subgraph Generative Visuals Loop
            D1[Prompt Analysis] --> D2[Imagen / Cloudflare];
            D2 --> D3{Critique?};
            D3 -- Improve --> D2;
            D3 -- OK --> D4[Veo Text-to-Video];
        end
        D --> E(6. Audio Gen + Transcription);
        E --> F(7. Asset Aggregation);
        F --> G(8. Video Assembly);
    end

    subgraph Final Output [✅ Current Stage]
        G --> H[Final Video (.mp4 on GCS)];
        H --> I[Display Video in UI];
    end

    subgraph Next Steps [➡️ Next Steps / Future]
        direction TB
        G --> J[Refine Assembly: Effects, Transitions];
        J --> K[Improve Synchronization Logic];
        E --> L[Explore Alternative TTS/STT for Cost/Quality];
        B2 --> M[Real-time Data Integration];
        I --> N[Alternative Outputs: Interactive Card, Article];
    end

    subgraph Long-Term Vision [🚀 Future Enhancements]
        K --> O[Advanced Editing / Automated Storytelling];
        M --> P[Deeper Stat Integration / Viz];
        N --> Q[User Customization / Style Control];
    end

    style Final Output fill:#d2f7d2,stroke:#333,stroke-width:2px
    style Next Steps fill:#d2e7f7,stroke:#333,stroke-width:2px
    style Long-Term Vision fill:#f7f7d2,stroke:#333,stroke-width:2px
"""

st.graphviz_chart(roadmap_diagram, use_container_width=True)


st.markdown("---")

# --- Detailed Future Plans ---
st.header("➡️ Next Steps & Future Plans")

st.subheader("Immediate Focus: Refinements & Robustness")
st.markdown("""
*   **Goal:** Improve the quality, reliability, and efficiency of the current video generation pipeline.
*   **Synchronization Accuracy:** Enhance the LLM-based visual timeline generation. Investigate potential issues causing mismatches between dialogue and visuals. Explore rule-based heuristics alongside the LLM.
*   **Video Assembly Quality:**
    *   Add basic transitions (e.g., crossfades) between visual segments in MoviePy instead of hard cuts.
    *   Implement simple effects for static images (e.g., Ken Burns effect - subtle pan/zoom) to make them more dynamic.
    *   Optimize `ffmpeg` parameters used by MoviePy for better compression/quality balance.
*   **Error Handling:** Improve error detection and reporting throughout the agent graph, especially in the assembly phase. Provide clearer feedback to the user if a specific step fails.
*   **Cost/Performance Analysis:** Evaluate the cost of different components (LLMs, Imagen, Veo, TTS, STT) and identify potential optimizations (e.g., using smaller models where appropriate, caching results).
*   **Prompt Engineering:** Continuously refine prompts for dialogue generation, visual generation (Imagen/Veo safety), and timeline mapping for better results.

""")

st.subheader("Mid-Term Goals")
st.markdown("""
*   **Alternative Output Formats:**
    *   **Interactive HTML/JS Card:** Develop a web component where the script text highlights dynamically as the audio plays, and the relevant visual appears alongside.
    *   **Automated Article Generation:** Adapt the agent to produce a written summary article with embedded images/videos.
*   **User Feedback Loop:** Implement a mechanism for users to rate or provide feedback on the generated videos, which could be used for future model fine-tuning or prompt adjustments.
*   **Expanded Data Sources:** Incorporate additional data points like betting odds, injury reports (via web search), or historical game comparisons.
""")


st.subheader("Long-Term Vision")
st.markdown("""
*   **Real-time Capability:** Integrate with live MLB data APIs (e.g., MLB Stats API feeds) for near real-time recap generation shortly after a game concludes.
*   **Deeper Analysis & Storytelling:** Incorporate more advanced stats (wOBA, FIP, Win Probability Added), automatically identify key narrative turning points, and potentially generate analytical graphics.
*   **Advanced Visual Editing:** Explore more complex video editing techniques, potentially integrating with cloud-based video editing APIs or generating more dynamic motion graphics.
*   **Customization & Control:** Allow users more control over the output style (e.g., host personalities, visual density), length, and specific players/plays to focus on.
*   **Multi-Platform Outputs:** Generate outputs optimized for different platforms (e.g., vertical video for TikTok/Shorts, concise summaries for Twitter).
""")

st.markdown("---")
st.caption("This roadmap is subject to change based on development progress, user feedback, and evolving AI capabilities.")