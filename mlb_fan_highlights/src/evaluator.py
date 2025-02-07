from IPython.display import HTML, Markdown, display
from google import genai
from google.genai.types import (
    GenerateContentConfig,
    GoogleSearch,
    Part,
    Retrieval,
    SafetySetting,
    Tool,
    VertexAISearch,
)
from google.cloud import bigquery
import os
import logging
from google.api_core import exceptions
from typing import List, Dict, Union, Optional
from datetime import datetime
import urllib.parse
import json


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


PROJECT_ID = "gem-rush-007"  # Replace with your actual Google Cloud project ID
os.environ["GOOGLE_CLOUD_PROJECT"] = PROJECT_ID # set this environment variable to your project ID
bq_client = bigquery.Client(project=PROJECT_ID)

client = genai.Client(vertexai=True, project="gem-rush-007", location="us-central1")
MODEL_ID = "gemini-2.0-pro-exp-02-05"  # @param {type: "string"}

safety_settings = [
    SafetySetting(
        category="HARM_CATEGORY_DANGEROUS_CONTENT",
        threshold="BLOCK_LOW_AND_ABOVE",
    ),
    SafetySetting(
        category="HARM_CATEGORY_HARASSMENT",
        threshold="BLOCK_LOW_AND_ABOVE",
    ),
    SafetySetting(
        category="HARM_CATEGORY_HATE_SPEECH",
        threshold="BLOCK_LOW_AND_ABOVE",
    ),
    SafetySetting(
        category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
        threshold="BLOCK_LOW_AND_ABOVE",
    ),
]




def evaluate_podcast_script(script: str, original_question: str) -> dict:
    """
    Evaluates a podcast script using Gemini, providing feedback on various aspects.

    Args:
        script (str): The JSON formatted podcast script to evaluate.
        original_question (str): The user's original prompt that was used to generate the script.

    Returns:
        dict: A dictionary containing the evaluation results.
    """
    client = genai.Client(vertexai=True, project="gem-rush-007", location="us-central1")
    MODEL_ID = "gemini-2.0-flash-exp"

    evaluator_prompt = f"""
    You are a highly skilled podcast script evaluator. Your task is to critically assess a given sports podcast script and provide detailed feedback. Consider the following aspects:

    **Evaluation Criteria:**

    1.  **Relevance to the Original Question:**
        *   How well does the script address the original user question?
        *   Does it focus on the key aspects requested, including specific teams, players, and timeframes?
        *   Does it effectively extract all relevant information from the user's question?
        *   Does the script address the question in a comprehensive manner?
        *   Is the scope of the script appropriate for the given user request?
        *   Rate the relevance of the script to the user's question on a scale from 1 to 10, where 1 is not relevant at all and 10 is extremely relevant.

    2.  **Data Accuracy and Completeness:**
        *   Is the data presented in the script accurate and supported by the available data?
        *   Are key game events, stats, and player information correctly presented?
        *   Are there any factual errors or missing pieces of crucial information?
        *  If some data was missing did the script appropriately call out that data was missing?
        *   Does the script use data to effectively enhance the script?
        * Rate the accuracy of the data on a scale of 1 to 10, where 1 indicates extremely inaccurate and 10 indicates perfectly accurate.

    3.  **Multi-Speaker Script Quality:**
        *   How effectively does the script utilize the three speaker roles (Play-by-play Announcer, Color Commentator, Simulated Player Quotes)?
        *   Is each speaker role distinct, with clear variations in language and tone?
        *   Does the script maintain a natural and engaging conversation flow between the speakers?
         * Does the script seamlessly transition between different play segments?
        *   Are simulated player quotes realistic and fitting for the context?
        *   Rate the overall quality of the multi-speaker script on a scale of 1 to 10, where 1 indicates poor use of speakers and 10 indicates excellent use of speakers.

    4.  **Script Flow and Coherence:**
        *   Does the script have a logical flow and a clear narrative structure?
        *   Are transitions between plays smooth and coherent?
        *   Does the script provide a good listening experience?
        *    Are the plays and events presented in a logical order?
        *   Rate the script's flow and coherence on a scale of 1 to 10, where 1 indicates disjointed and 10 indicates a perfectly coherent flow.

    5. **Use of Edge Case Logic:**
        * Does the script appropriately handle edge cases such as data gaps or unexpected situations?
        *   Does the script fail gracefully when faced with missing or incomplete data?
        *    Does the script accurately report any error conditions?
        *    Rate the edge case handling of the script on a scale of 1 to 10, where 1 indicates poor handling and 10 indicates excellent handling.

    **Output:**
    *   Provide a detailed evaluation report. For each criterion provide a score from 1-10, where 1 is the lowest score and 10 is the highest score. Provide a summary paragraph explaining the evaluation for each category and the rational for why you rated each category as you did.
     * Format the output as a valid JSON object.

    **Input:**
        *   Original User Question: {original_question}
        *   Podcast Script: {script}

    Respond with a valid JSON object.
    """


    try:
        google_search_tool = Tool(google_search=GoogleSearch())
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=evaluator_prompt,
            config=GenerateContentConfig(
                temperature=0.2,
                top_p=0.95,
                top_k=40,
                max_output_tokens=2048,
                tools=[google_search_tool],
                safety_settings=safety_settings,
            ),
        )
        try:
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
            return text_response
        except json.JSONDecodeError as e:
             logging.error(f"JSON Decode Error in evaluate_podcast_script: {e}, response was {text}")
             return {
                "error": f"JSON Decode Error in evaluate_podcast_script: {e}, please check the logs"
            }
    except Exception as e:
        logging.error(f"Error in evaluate_podcast_script: {e}")
        return {
            "error": f"An error occurred: {e}",
        }
    
