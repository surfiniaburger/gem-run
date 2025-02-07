from typing import Dict, List, Optional
from vertexai.language_models import TextEmbeddingModel
from pymongo.mongo_client import MongoClient
import logging
import google.cloud.logging
from google.cloud.logging.handlers import CloudLoggingHandler
from google.cloud import secretmanager
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from google.cloud import bigquery
from IPython.display import HTML, Markdown, display
import google.generativeai as genai
import os
from google.api_core import exceptions
from typing import List, Dict, Union, Optional
from datetime import datetime
import urllib.parse
import json
from evaluator import evaluate_podcast_script


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


PROJECT_ID = "gem-rush-007"  # Replace with your actual Google Cloud project ID
os.environ["GOOGLE_CLOUD_PROJECT"] = PROJECT_ID # set this environment variable to your project ID
bq_client = bigquery.Client(project=PROJECT_ID)

#client = genai.Client(vertexai=True, project="gem-rush-007", location="us-central1")
#MODEL_ID = "gemini-2.0-pro-exp-02-05"  # @param {type: "string"}




# --- Setup (Logging, Secret Manager, BigQuery Client) ---

def setup_logging():
    """Sets up Google Cloud Logging."""
    client = google.cloud.logging.Client()
    handler = CloudLoggingHandler(client)
    logger = logging.getLogger('mongodb_vector_search')
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    return logger

logger = setup_logging()

def get_secret(project_id, secret_id, version_id="latest", logger=None):
    """Retrieves a secret from Google Cloud Secret Manager."""
    try:
        client = secretmanager.SecretManagerServiceClient()
        name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"
        response = client.access_secret_version(request={"name": name})
        return response.payload.data.decode("UTF-8")
    except Exception as e:
        if logger:
            logger.error(f"Failed to retrieve secret {secret_id}: {str(e)}")
        raise

# Assuming you have your Google Cloud project ID set as an environment variable
PROJECT_ID = "gem-rush-007"  # Replace with your actual project ID
bq_client = bigquery.Client(project=PROJECT_ID)
secret_id = "GEM-RUN-API-KEY"
apiKey = get_secret(PROJECT_ID, secret_id, logger=logger)

if apiKey:
  GOOGLE_API_KEY = apiKey
  genai.configure(api_key=GOOGLE_API_KEY)
  

TEAMS = {
    'rangers': 140,
    'angels': 108,
    'astros': 117,
    'rays': 139,
    'blue_jays': 141,
    'yankees': 147,
    'orioles': 110,
    'red_sox': 111,
    'twins': 142,
    'white_sox': 145,
    'guardians': 114,
    'tigers': 116,
    'royals': 118,
    'padres': 135,
    'giants': 137,
    'diamondbacks': 109,
    'rockies': 115,
    'phillies': 143,
    'braves': 144,
    'marlins': 146,
    'nationals': 120,
    'mets': 121,
    'pirates': 134,
    'cardinals': 138,
    'brewers': 158,
    'cubs': 112,
    'reds': 113,
    'athletics': 133,
    'mariners': 136,
    'dodgers': 119,
}

# Dictionary to map full team names to their corresponding keys in TEAMS
FULL_TEAM_NAMES = {
    'texas rangers': 'rangers',
    'los angeles angels': 'angels',
    'houston astros': 'astros',
    'tampa bay rays': 'rays',
    'toronto blue jays': 'blue_jays',
    'new york yankees': 'yankees',
    'baltimore orioles': 'orioles',
    'boston red sox': 'red_sox',
    'minnesota twins': 'twins',
    'chicago white sox': 'white_sox',
    'cleveland guardians': 'guardians',
    'detroit tigers': 'tigers',
    'kansas city royals': 'royals',
    'san diego padres': 'padres',
    'san francisco giants': 'giants',
    'arizona diamondbacks': 'diamondbacks',
    'colorado rockies': 'rockies',
    'philadelphia phillies': 'phillies',
    'atlanta braves': 'braves',
    'miami marlins': 'marlins',
    'washington nationals': 'nationals',
    'new york mets': 'mets',
    'pittsburgh pirates': 'pirates',
    'st louis cardinals': 'cardinals',
    'milwaukee brewers': 'brewers',
    'chicago cubs': 'cubs',
    'cincinnati reds': 'reds',
    'oakland athletics': 'athletics',
    'seattle mariners': 'mariners',
    'los angeles dodgers': 'dodgers',
}


def get_team_key(team_name: str) -> Optional[str]:
    """Retrieves the team key (short name) from a team name."""
    team_name = team_name.lower().strip()
    if team_name in TEAMS:
        return team_name
    if team_name in FULL_TEAM_NAMES:
        return FULL_TEAM_NAMES[team_name]
    for full_name, short_name in FULL_TEAM_NAMES.items():
        if team_name in full_name:
            return short_name
    for short_name in TEAMS:
        if team_name in short_name:
            return short_name
    return None




def query_mongodb_for_agent(
    query_text: str, db_name: str = "mlb_data", limit: int = 5
) -> list:
    """
    Queries MongoDB using vector similarity, extracting the team name.

    This function takes a user's query, converts it to an embedding, searches
    the appropriate MongoDB collection for similar play descriptions, and
    returns the results in a structured format suitable for the agent to process.
    It handles the team-specific collection logic.

    Args:
        query_text: The user's query string.
        db_name: The name of the MongoDB database (default: "mlb_data").
        limit: The maximum number of results (default: 5).

    Returns:
        A list of dictionaries, or an empty list on error/no results.
    """
    # --- Initialize MongoClient INSIDE the function ---
    try:
        project_id = "gem-rush-007"  # Replace with your project ID
        secret_id = "mongodb-uri"
        uri = get_secret(project_id, secret_id, logger=logger)
        client = MongoClient(uri, server_api=ServerApi('1'))
        client.admin.command('ping')  # Test connection
        logger.info("Successfully connected to MongoDB within the function!")
    except Exception as e:
        logger.error(f"Error connecting to MongoDB: {e}")
        return []  # Return empty list on connection failure

    model = TextEmbeddingModel.from_pretrained("text-embedding-005")
    try:
        query_embedding = model.get_embeddings([query_text])[0].values
    except Exception as e:
        logger.error(f"Failed to generate embedding: {e}")
        client.close() #close the client
        return []

    if not query_embedding:
        logger.error("Failed to generate embedding")
        client.close() #close the client
        return []

    db = client[db_name]

    # --- Extract Team Name ---
    team_key = None
    for team in list(TEAMS.keys()) + list(FULL_TEAM_NAMES.keys()):
        if team.lower() in query_text.lower():
            team_key = get_team_key(team)
            break

    if team_key:
        collection_name = f"{team_key}_plays"
        logger.info(f"Querying collection: {collection_name}")
    else:
        collection_name = "all_plays" # not yet implemented, due to cluster constraint
        logger.info(f"No team specified, querying: {collection_name}")

    collection = db[collection_name]
    if collection.count_documents({}) == 0:
        logger.warning(f"Collection '{collection_name}' is empty/missing.")
        client.close() #close the client
        return []

    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",
                "path": "embedding",
                "queryVector": query_embedding,
                "numCandidates": 150,
                "limit": limit,
            }
        },
        {
            "$project": {
                "_id": 0,
                "official_date": 1,
                "game_id": 1,
                "home_team_name": 1,
                "away_team_name": 1,
                "description": 1,
                "event": 1,
                "batter_name": 1,
                "pitcher_name": 1,
                "inning": 1,
                "half_inning": 1,
                "rbi": 1,
                "is_scoring_play": 1,
                "score": {"$meta": "vectorSearchScore"},
            }
        },
    ]

    try:
        results = list(collection.aggregate(pipeline))
        return results
    except Exception as e:
        logger.error(f"Error querying MongoDB: {e}")
        return []
    finally:
        client.close()  # ALWAYS close the connection when done


# teams available now include rays, rangers, astros, angels


def generate_mlb_podcasts(contents: str) -> dict:
    MODEL_ID = "gemini-2.0-pro-exp-02-05"

    # Structure the prompt to explicitly request tool usage
    structured_prompt = f"""
    You are an expert sports podcast script generator, adept at creating engaging, informative, and dynamic scripts based on user requests and available data. Your task is multifaceted, requiring precise execution across several stages to ensure exceptional output.

    **Overall Goal:** To produce a compelling and meticulously crafted podcast script that accurately addresses user requests, leverages available data effectively, and provides a high-quality listening experience.

    **Step 1: Comprehensive User Request Analysis**
        *   **In-Depth Scrutiny:**  Thoroughly examine the "Question" field, extracting all explicit and implicit requirements. This includes:
            *   **Specificity:** Identify all mentioned teams, players, games (or specific time periods).
            *   **Game Context:** Determine the game type (e.g., regular season, playoffs, exhibition), any specific game focus (key plays, player performance), and critical moments (turning points, upsets).
            *   **Content Focus:** Pinpoint the desired podcast focus (e.g., game analysis, player highlights, team strategy, historical context, record-breaking events).
            *   **Stylistic Preferences:** Understand the desired podcast tone and style (e.g., analytical, enthusiastic, humorous, serious, historical, dramatic).
            *    **Statistical Emphasis:** Identify any specific stats, metrics, or data points the user wants to highlight, including, but not limited to, game dates, final scores, player specific metrics, and any other metrics that provide greater depth to the game. **Crucially, prioritize including all available statistics for mentioned players, teams, and their opponents. This should include, but is not limited to, batting averages, home runs, RBIs, pitching stats (ERA, strikeouts, wins/losses), and fielding statistics. Additionally, be sure to include the names of all starting and key relief pitchers for the game.**
            *   **Implicit Needs:** Infer unspoken requirements based on the question's context (e.g., if a user asks about a close game, anticipate a focus on the final moments).
        *   **Data Prioritization Logic:**  Establish a clear hierarchy for data based on user needs. For example:
            *   Player-centric requests: Prioritize individual player stats, highlights, and pivotal moments.
            *   Game-focused requests: Prioritize game summaries, key events, and strategic plays.
            *   Historical requests: Focus on past game data, trends, records, and historical context.
        *   **Edge Case Management:** Implement robust logic to manage varied user inputs. Specifically:
            *   **Vague Queries:** Develop a fallback strategy for questions like "Tell me about the Lakers." Provide a balanced overview that includes recent games, important historical moments, and significant player performances.
            *   **Conflicting Directives:**  Create a resolution strategy for contradictory requirements (e.g., focus on Player A and Team B). Balance the requests or prioritize based on a logical interpretation of the question. Highlight points where those focus areas intersect in an organic way.
            - **Data Gaps:** If specific game data (e.g., game dates, final scores, **player stats**, , **pitcher information**) is missing, explicitly state in the script that the data was unavailable. Do not use placeholder values. 
            *  **Off-Topic Inquiries:** If the request falls outside the tool's scope (e.g., "What does player X eat"), acknowledge the request is out of scope with a concise message.
            *   **Multiple Entities:** If the user asks for information on multiple teams or players, without specifying a game, provide a summary of their recent performances.
            *  **Aggregated Data:** If the user requests a summary or comparison of multiple players across multiple games, generate an aggregated summary for each player across those games.
            *  **Canceled Events:** If the user requests a game that did not happen, then acknowledge the cancellation.

    **Step 2: Strategic Data Acquisition and Intelligent Analysis**
        *   **Dynamic Tool Selection:** Select the most suitable tool(s) from the available resources based on the refined needs identified in Step 1.  Tools can include statistical APIs, play-by-play logs, news feeds, and social media. Use multiple tools if necessary to gather all the necessary information.
        *  **Prioritized Data Retrieval:** If past games are requested, treat these as primary sources of data and emphasize those data sets. If the user requests a future game or a game with no available data, then state that explicitly in the generated text and use available information like team projections, past performance or other pre game analysis information.
        *   **Granular Data Extraction:** Extract relevant data points, focusing on:
            *   **Critical Events:** Highlight game-changing plays (e.g., game-winning shots, home runs, interceptions).
            *   **Performance Extremes:** Note exceptional performances, unusual dips in performance, or record-breaking accomplishments.
            *   **Pivotal Moments:**  Identify turning points that altered the course of the game.
            *   **Player Insight:** Analyze and report on detailed player actions, individual statistics, and contributions to the game. **Include all relevant stats, such as batting average, home runs, RBIs, and any other available metrics.**
            *   **Game Details:** Extract and include game dates, final scores, and any other relevant game details that add depth and context to the discussion.
            *    **Pitcher Information:** Include starting and key relief pitcher names for each team, as well as their individual stats for the game where available (e.g., innings pitched, strikeouts, earned runs).
        *  **Contextual Layering:** Augment raw data with contextual information to enrich the analysis.
        *  **Contextual Layering:** Augment raw data with contextual information to enrich the analysis.
            *    **Historical Data:** Use past data, historical performance, and historical records, team or player-specific trends to provide the analysis greater depth.
            *    **Team Specific Data:** Use team specific data to better inform the analysis (e.g. if a team is known for strong defense, then analyze this and provide commentary on it).
        *  **Data Integrity Checks:** Sanitize the data to ensure only relevant information is extracted from all sources. Clean and remove any unwanted data.
        * **Edge Case Resolution:** Implement rules for specific edge cases:
            *   **Incomplete Data:** If data is missing or incomplete, explicitly mention this within the generated text using phrases like "data was not available for this event."
            *   **Data Conflicts:** Prioritize reliable sources. If discrepancies persist, note these in the generated text. Explain differences, and any issues that may exist in the data.
            *  **Data Format Issues:**  If the data cannot be parsed or used, then log a detailed error and provide the user with an error in the generated text that explains why data was not used. If possible perform data transformations.

    **Step 3: Advanced Multi-Speaker Script Composition**
        *   **Speaker Profiles:** Develop unique personality profiles for each speaker role to ensure variations in voice and perspective:
             *   **Play-by-play Announcer:** Neutral, factual, and descriptive, providing real-time action updates using clear language.
            *   **Color Commentator:** Analytical, insightful, and contextual, breaking down game elements, offering explanations, and using phrases like "what's interesting here is," "the reason why," and "a key moment in the game".
            *   **Simulated Player Quotes:** Casual, personal, and engaging, re-creating player reactions with plausible, authentic-sounding phrases. **Ensure that for each key play, a simulated player quote is present, that is relevant to the play and provides a unique perspective on the action.**
        *   **Event-Driven Structure:** Structure the script around the key events identified in Step 2. For each event:
             *   Involve all three speaker roles in the conversation to provide multiple perspectives.
            *   Maintain a natural conversation flow, resembling a genuine podcast format.
            *   Incorporate all available relevant information, including player names, team names, inning details, and applicable statistics, **game dates and final scores, and player and pitcher specific stats.**.
        *   **Seamless Transitions:** Use transitional phrases (e.g., "shifting to the next play," "now let's look at the defense") to ensure continuity.
        *   **Unbiased Tone:** Maintain a neutral and factual tone, avoiding any personal opinions, unless specifically instructed by the user.
        *   **Edge Case Handling:**
            *   **Tone Alignment:** Ensure that the speaker's tone reflects the events described (e.g., use a negative tone for the color commentator if describing a poorly executed play).
            *   **Quote Realism:** Ensure simulated quotes are believable and sound authentic.
            *   **Data Gaps:** If there's missing data or an unexpected scenario, use filler phrases (e.g., "We don't have the audio for that play," "Unfortunately, the camera wasn't on the ball").

    **Step 4: Globally Accessible Language Support**
        *   **Translation Integration:** Use translation tools to translate the full output, including all generated text, data-driven content, and speaker roles.
        *   **Language-Specific Adjustments and Chain of Thought Emphasis:**
              - **For Japanese:**  
                   • Use culturally appropriate sports broadcasting language.  
                   • Emphasize the inclusion of the game date and final score by using precise Japanese conventions. 
                   • **Chain-of-Thought:** Begin by clearly stating the game date using Japanese date formats (e.g., "2024年5月15日") and then present the final score using phrases such as "最終スコア." Anchor the entire script in these key details to build a solid factual framework. As you proceed, refer back to these details when transitioning between segments, ensuring that every pivotal play is contextualized within the exact game date and score. This approach not only reinforces the factual basis of the narrative but also resonates with Japanese audiences who expect precision and clarity in sports reporting.
              - **For Spanish:**  
                   • Adopt a lively and engaging commentary style typical of Spanish sports media.  
                   • Stress the inclusion of the game date and final score by using phrases like "la fecha del partido" and "el marcador final" to provide clear factual anchors.  
                   • Chain of Thought: Start the script by emphasizing the importance of the game date using spanish date format and final score, setting the stage for a dynamic narrative. Use vivid descriptions and energetic language to draw the listener into the game, making sure to highlight these key data points repeatedly throughout the script to reinforce the factual context. Detailed descriptions of pivotal plays and smooth transitions will maintain listener engagement while ensuring that the essential facts are always in focus.
              - **For English:**  
                   • Maintain the current detailed and structured narrative with clear emphasis on game dates and final scores as factual anchors.
        *  **Default Language Protocol:** If the user does not specify a language, English will be used as the default language.
        *   **Translation Quality Assurance:** Verify that the translation is accurate and reflects the intended meaning. Ensure that the context of the original text is not lost in translation.
        *   **Edge Case Adaptations:**
            *   **Incomplete Translations:** If the translation is incomplete, use an error code for that section (e.g., `[translation error]`).
            *   **Bidirectional Languages:** Handle languages that read right-to-left to ensure proper text rendering.
           *  **Contextual Accuracy:** Ensure the translation maintains the appropriate tone for the speakers.

    **Step 5: Structured JSON Output Protocol**
        *   **JSON Formatting:** Create the output as a valid JSON array without any additional formatting.
        *   **Speaker and Text Fields:** Each JSON object must include two fields: `"speaker"` and `"text"`.
        *   **Single Array Format:** The output must be a single JSON array containing the entire script.
        *   **No Markdown or Code Blocks:** Do not include any markdown or other formatting elements.
        *   **JSON Validation:** Validate that the output is proper JSON format prior to output.
         *  **Example JSON:**
            ```json
            [
                {{
                    "speaker": "Play-by-play Announcer",
                    "text": "Here's the pitch, swung on and a long drive..."
                }},
                {{
                    "speaker": "Color Commentator",
                    "text": "Unbelievable power from [Player Name] there, that was a no doubter."
                }},
                {{
                    "speaker": "Player Quotes",
                    "text": "I knew I was gonna hit that out of the park!"
                }}
            ]
            ```
        *   **Edge Case Management:**
            *   **JSON Errors:** If there is a problem creating the json object, then return a json object with an error message.
    **Your Output must be a pure JSON array without any markdown code blocks or formatting. Just the raw JSON.**

    Question: {contents}

    Prioritize the correct execution of each step to ensure the creation of a high-quality, informative, and engaging podcast script, fully tailored to the user's request. Be sure to consider any edge cases in the process.
    """
    # Create chat session with automatic function calling
    model = genai.GenerativeModel(
        model_name=MODEL_ID,
        tools=[query_mongodb_for_agent],  # Directly pass the function
        system_instruction=structured_prompt
    )
    chat = model.start_chat(enable_automatic_function_calling=True) 
    try:
        response = chat.send_message(contents)

        try:
            # Clean the response text by removing markdown code block syntax
            text = response.text
            if text.startswith("```"):
                # Find the first newline after the opening ```
                start_idx = text.find("\n") + 1
                # Find the last ``` and exclude everything after it
                end_idx = text.rfind("```")
                if end_idx == -1:  # If no closing ```, just remove the opening
                    text = text[start_idx:]
                else:
                    text = text[start_idx:end_idx].strip()
            
            # Remove any "json" or other language identifier that might appear
            text = text.replace("json\n", "")
            
            # Parse the cleaned JSON
            text_response = json.loads(text)
            evaluation = evaluate_podcast_script(text, contents)
            print(evaluation)
            print(text_response)
            return text_response
        except json.JSONDecodeError as e:
            logging.error(f"JSON Decode Error in generate_mlb_analysis: {e}, response was {text}")
            return {
                "error": f"JSON Decode Error in generate_mlb_analysis: {e}, please check the logs"
            }
    except Exception as e:
        logging.error(f"Error in generate_mlb_analysis: {e}")
        return {
            "error": f"An error occurred: {e}",
        }



# Add this helper function for processing responses
def process_chat_response(history):
    """Extracts the final response from chat history"""
    for content in reversed(history):
        if content.role == "model":
            for part in content.parts:
                if part.text:
                    return clean_json_response(part.text)
    return {"error": "No valid response found"}

def clean_json_response(text):
    """Cleans and parses JSON response"""
    clean_text = text.strip().replace("```json\n", "").replace("```", "")
    try:
        return json.loads(clean_text)
    except json.JSONDecodeError:
        logging.error(f"Failed to parse JSON response: {clean_text}")
        return {"error": "Invalid JSON format in response"}

generate_mlb_podcasts("Show me recent plays by angels")
query_mongodb_for_agent("Show me recent plays by angels")
print(query_mongodb_for_agent("Show me recent plays by angels"))