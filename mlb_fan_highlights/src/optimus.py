# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


PROJECT_ID = "gem-rush-007"  # YOUR PROJECT ID
LOCATION = "us-central1"
STAGING_BUCKET = "gs://gem-rush-007-reasoning-engine"  # YOUR BUCKET

import vertexai
vertexai.init(project=PROJECT_ID, location=LOCATION, staging_bucket=STAGING_BUCKET)

# --- Install Dependencies (as in the template) ---
# %pip install --upgrade --quiet \
#     "google-cloud-aiplatform[langchain,reasoningengine]" \
#     cloudpickle==3.0.0 \
#     "pydantic>=2.10" \
#     requests \
#     google-cloud-bigquery \
#     google-cloud-secret-manager \
#     google-cloud-logging\
#     pymongo


# --- Define Model ---
model = "gemini-2.0-pro-exp-02-05"  # Start with 1.5-pro for reliability

# --- Define Tools (adapted functions from mlb_data_functions.py) ---
# IMPORTANT:  These are the *adapted* versions, as described above.
from mlb_data_functions import (
    fetch_team_games,
    fetch_team_player_stats,
    fetch_team_player_stats_by_opponent,
    fetch_team_player_stats_by_game_type,
    fetch_team_plays,
    fetch_team_plays_by_opponent,
    fetch_team_plays_by_game_type,
    fetch_team_games_by_opponent,
    fetch_team_games_by_type,
    fetch_player_game_stats,
    fetch_player_plays,
    fetch_player_plays_by_opponent,
    fetch_player_plays_by_game_type
)
from vertexai.preview import reasoning_engines

# --- Define Agent ---
agent = reasoning_engines.LangchainAgent(
    model=model,
    tools=[
        fetch_team_games,
        fetch_team_player_stats,
        fetch_team_player_stats_by_opponent,
        fetch_team_player_stats_by_game_type,
        fetch_team_plays,
        fetch_team_plays_by_opponent,
        fetch_team_plays_by_game_type,
        fetch_team_games_by_opponent,
        fetch_team_games_by_type,
        fetch_player_game_stats,
        fetch_player_plays,
        fetch_player_plays_by_opponent,
        fetch_player_plays_by_game_type,
        # ... include all your adapted data functions ...
    ],
    agent_executor_kwargs={"return_intermediate_steps": True},
)

# --- Test Locally (IMPORTANT!) ---
print("Testing locally...")
test_query = "What were the results of the last two Rangers games?"
local_response = agent.query(input=test_query)
print(f"Local Response: {local_response}")

# for local_chunk in agent.stream_query(input=test_query):
#   print(local_chunk)


# --- Deploy to Vertex AI ---
print("Deploying to Vertex AI...")
remote_agent = reasoning_engines.ReasoningEngine.create(
    agent,
    requirements=[
        "google-cloud-aiplatform[langchain,reasoningengine]",
        "cloudpickle==3.0.0",
        "pydantic>=2.10",
        "requests",
        "google-cloud-bigquery",
        "google-cloud-secret-manager",
        "google-cloud-logging",
        "pymongo",
        "urllib3",
        # List ALL your dependencies
    ],
)

print(f"Deployed Reasoning Engine: {remote_agent.resource_name}")

# --- Test Remotely ---
print("Testing remotely...")
remote_response = remote_agent.query(input=test_query)
print(f"Remote Response: {remote_response}")

# for remote_chunk in remote_agent.stream_query(input=test_query):
#   print(remote_chunk)

# --- Example of how to use the deployed agent from another script/notebook ---
# print("Example usage from another script/notebook:")
# print(f"""
# from vertexai.preview import reasoning_engines

# REASONING_ENGINE_RESOURCE_NAME = "{remote_agent.resource_name}"  # Use the resource name

# remote_agent = reasoning_engines.ReasoningEngine(REASONING_ENGINE_RESOURCE_NAME)
# response = remote_agent.query(input="Give me a summary of the last Yankees game.")
# print(response)
# """)

# --- Clean Up (when done) ---
# remote_agent.delete() # Uncomment when ready to delete

# --- generate_mlb_podcasts (Example usage) ---
# Now, in a separate part of your notebook (or in a different script), you would use
# your `generate_mlb_podcasts` function, which interacts with the *deployed* agent.
# You do NOT deploy `generate_mlb_podcasts` itself.
#
# from your_app_module import generate_mlb_podcasts  # Assuming it's in your_app_module.py

# podcast_script = generate_mlb_podcasts("Create a podcast script about the last Rangers game.")
# print(podcast_script)